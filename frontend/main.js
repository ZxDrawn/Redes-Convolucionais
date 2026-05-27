import * as tf from '@tensorflow/tfjs';

// ============================================================
// ARQUITETURA: Transfer Learning com MobileNet
// O MobileNet foi treinado no ImageNet (14 milhões de imagens).
// Usamos ele apenas como "extrator de características" (features),
// congelamos seus pesos e treinamos só a camada final com as
// suas fotos. Isso é exatamente o que o Google Teachable Machine faz.
// ============================================================

const IMAGE_SIZE = 224; // Tamanho que o MobileNet espera
let mobileNet = null;   // Rede pré-treinada (extrator de features)
let classifier = null;  // Nossa camada de classificação (treinável)

let class1Embeddings = []; // Vetores de features extraídos das imagens da classe 1
let class2Embeddings = []; // Vetores de features extraídos das imagens da classe 2
let class1Images = [];
let class2Images = [];
let testImageTensor = null;

// ---- UI Elements ----
const uploadClass1 = document.getElementById('uploadClass1');
const btnClass1 = document.getElementById('btnClass1');
const previewClass1 = document.getElementById('previewClass1');
const countClass1 = document.getElementById('countClass1');

const uploadClass2 = document.getElementById('uploadClass2');
const btnClass2 = document.getElementById('btnClass2');
const previewClass2 = document.getElementById('previewClass2');
const countClass2 = document.getElementById('countClass2');

const trainBtn = document.getElementById('trainBtn');
const trainingStatus = document.getElementById('trainingStatus');
const trainProgress = document.getElementById('trainProgress');
const trainProgressFill = document.getElementById('trainProgressFill');
const lossText = document.getElementById('lossText');

const uploadTest = document.getElementById('uploadTest');
const btnTest = document.getElementById('btnTest');
const runInferenceBtn = document.getElementById('runInferenceBtn');
const canvasInput = document.getElementById('canvasInput');
const networkVisualizer = document.getElementById('networkVisualizer');

const outputContainers = {
  conv: document.getElementById('convOutputs'),
  pool: document.getElementById('poolOutputs'),
  class: document.getElementById('classOutputs')
};

// ============================================================
// PASSO 0: Carregar o MobileNet ao iniciar a página
// ============================================================
async function loadMobileNet() {
  trainingStatus.textContent = 'Carregando MobileNet pré-treinado...';
  // Carregamos o MobileNet e extraímos um sub-modelo que termina
  // na camada 'global_average_pooling2d_1' — o ponto que gera o
  // "vetor de características" de 1280 dimensões de cada imagem.
  const mobilenet = await tf.loadLayersModel(
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
  );
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  mobileNet = tf.model({ inputs: mobilenet.inputs, outputs: layer.output });

  trainingStatus.textContent = 'MobileNet pronto! Envie suas imagens nas duas classes.';
  trainingStatus.style.color = '#34d399';
  btnClass1.disabled = false;
  btnClass2.disabled = false;
}
loadMobileNet();

// ============================================================
// PRÉ-PROCESSAMENTO
// ============================================================
function preprocessImage(img) {
  return tf.tidy(() => {
    return tf.browser.fromPixels(img)
      .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE]) // MobileNet exige 224x224
      .toFloat()
      .div(127.5)
      .sub(1)                                  // Normalização: -1 a 1 (padrão MobileNet)
      .expandDims(0);                          // [1, 224, 224, 3]
  });
}

// Extrai o "vetor de características" de uma imagem usando o MobileNet congelado
function extractFeatures(img) {
  return tf.tidy(() => {
    const preprocessed = preprocessImage(img);
    return mobileNet.predict(preprocessed); // shape: [1, 7, 7, 256]
  });
}

// ============================================================
// COLETA DE DADOS
// ============================================================
function handleUpload(input, imageArray, embeddingArray, previewContainer, countElement) {
  input.addEventListener('change', async (e) => {
    const files = e.target.files;
    for (let file of files) {
      const img = await loadImage(file);
      imageArray.push(img);

      // Já extrai e guarda o vetor de features (mais rápido no treino)
      const embedding = extractFeatures(img);
      embeddingArray.push(embedding);

      const previewNode = document.createElement('img');
      previewNode.src = URL.createObjectURL(file);
      previewNode.className = 'image-preview';
      previewContainer.appendChild(previewNode);
    }
    countElement.textContent = `${imageArray.length} imagens carregadas`;
    checkReadyToTrain();
  });
}

function loadImage(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
}

btnClass1.addEventListener('click', () => uploadClass1.click());
btnClass2.addEventListener('click', () => uploadClass2.click());
handleUpload(uploadClass1, class1Images, class1Embeddings, previewClass1, countClass1);
handleUpload(uploadClass2, class2Images, class2Embeddings, previewClass2, countClass2);

function checkReadyToTrain() {
  if (class1Images.length > 0 && class2Images.length > 0) {
    trainBtn.disabled = false;
    trainingStatus.textContent = `Pronto! ${class1Images.length} imagens na Classe Alvo e ${class2Images.length} em Outros.`;
    trainingStatus.style.color = '#34d399';
  }
}

// ============================================================
// TREINAMENTO
// Apenas a camada densa final é treinada. Os pesos do MobileNet
// permanecem CONGELADOS. Isso é Transfer Learning.
// ============================================================
trainBtn.addEventListener('click', async () => {
  trainBtn.disabled = true;
  btnClass1.disabled = true;
  btnClass2.disabled = true;
  trainingStatus.textContent = 'Construindo classificador...';
  trainingStatus.style.color = '#f8fafc';

  // Descobre o shape do vetor de features gerado pelo MobileNet
  const sampleShape = class1Embeddings[0].shape.slice(1); // ex: [7, 7, 256]

  // Classificador: apenas estas camadas são treinadas do zero
  classifier = tf.sequential();
  classifier.add(tf.layers.flatten({ inputShape: sampleShape }));
  classifier.add(tf.layers.dense({ units: 64, activation: 'relu', name: 'hidden_1' }));
  classifier.add(tf.layers.dropout({ rate: 0.3 })); // Evita memorização dos exemplos
  classifier.add(tf.layers.dense({ units: 2, name: 'dense_out' })); // Logits
  classifier.add(tf.layers.softmax({ name: 'softmax_out' }));

  classifier.compile({
    optimizer: tf.train.adam(0.0005),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  // Montar o dataset a partir dos embeddings já extraídos
  const allEmbeddings = [...class1Embeddings, ...class2Embeddings];
  const labels = [
    ...class1Images.map(() => [1, 0]),
    ...class2Images.map(() => [0, 1])
  ];

  const xs = tf.tidy(() => tf.concat(allEmbeddings.map(e => e), 0)); // [N, 7, 7, 256]
  const ys = tf.tensor2d(labels, [labels.length, 2]);

  trainProgress.classList.remove('hidden');
  lossText.classList.remove('hidden');
  trainingStatus.textContent = 'Treinando classificador sobre os features do MobileNet...';

  const epochs = 150;
  await classifier.fit(xs, ys, {
    epochs,
    batchSize: Math.max(1, Math.floor(allEmbeddings.length / 2)),
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const percent = ((epoch + 1) / epochs) * 100;
        trainProgressFill.style.width = `${percent}%`;
        lossText.textContent = `Época ${epoch + 1}/${epochs} | Loss: ${logs.loss.toFixed(4)} | Acurácia: ${(logs.acc * 100).toFixed(1)}%`;
      }
    }
  });

  xs.dispose();
  ys.dispose();

  trainingStatus.textContent = '✅ Treinamento concluído!';
  trainingStatus.style.color = '#34d399';
  btnTest.disabled = false;
});

// ============================================================
// INFERÊNCIA
// ============================================================
btnTest.addEventListener('click', () => uploadTest.click());

uploadTest.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const img = await loadImage(file);

  const ctx = canvasInput.getContext('2d');
  ctx.clearRect(0, 0, canvasInput.width, canvasInput.height);
  const scale = Math.min(canvasInput.width / img.width, canvasInput.height / img.height);
  const w = img.width * scale;
  const h = img.height * scale;
  ctx.drawImage(img, (canvasInput.width - w) / 2, (canvasInput.height - h) / 2, w, h);

  if (testImageTensor) testImageTensor.dispose();
  testImageTensor = preprocessImage(img); // [1, 224, 224, 3]
  // Guarda o embedding para inferência
  testImageTensor._rawImg = img;

  runInferenceBtn.disabled = false;
  networkVisualizer.classList.add('hidden');
});

runInferenceBtn.addEventListener('click', async () => {
  if (!classifier || !testImageTensor) return;
  runInferenceBtn.disabled = true;
  networkVisualizer.classList.remove('hidden');

  // Extrair features da imagem de teste via MobileNet
  const testEmbedding = mobileNet.predict(testImageTensor); // [1, 7, 7, 256]

  // Interceptar os Logits brutos (antes do Softmax)
  const denseOutLayer = classifier.getLayer('dense_out');
  const logitsModel = tf.model({ inputs: classifier.inputs, outputs: denseOutLayer.output });

  const logitsTensor = logitsModel.predict(testEmbedding);
  const finalOut = classifier.predict(testEmbedding);

  const logitsData = await logitsTensor.data();
  const probsData = await finalOut.data();

  // Visualizar os mapas de ativação do MobileNet (primeiros 4 canais do embedding)
  await renderActivationMaps(testEmbedding, outputContainers.conv, 4);

  // Pooling visual: reduzir o embedding pela metade (simulando max pooling)
  const pooled = tf.tidy(() => tf.maxPool(testEmbedding, 2, 2, 'valid'));
  await renderActivationMaps(pooled, outputContainers.pool, 4);
  pooled.dispose();

  await renderClassBars(finalOut);
  renderMathExplanation(logitsData, probsData);

  tf.dispose([testEmbedding, logitsTensor, finalOut]);
  runInferenceBtn.disabled = false;
});

// ============================================================
// RENDERIZAÇÃO
// ============================================================
async function renderActivationMaps(tensor, container, maxFilters = 4) {
  container.innerHTML = '';
  const shape = tensor.shape; // [1, H, W, C]
  const numFilters = Math.min(shape[3], maxFilters);

  const min = tensor.min();
  const max = tensor.max();
  const normalized = tensor.sub(min).div(max.sub(min).add(1e-5));
  const unstacked = tf.unstack(normalized, 3);

  for (let i = 0; i < numFilters; i++) {
    const filterTensor = unstacked[i].squeeze([0]);
    const rgbTensor = filterTensor.mul(255).cast('int32').expandDims(-1).tile([1, 1, 3]);
    const canvas = document.createElement('canvas');
    canvas.width = shape[2];
    canvas.height = shape[1];
    await tf.browser.toPixels(rgbTensor, canvas);

    const wrap = document.createElement('div');
    wrap.className = 'canvas-wrapper';
    wrap.style.margin = '0';
    wrap.appendChild(canvas);
    container.appendChild(wrap);

    rgbTensor.dispose();
    filterTensor.dispose();
  }
  normalized.dispose();
  min.dispose();
  max.dispose();
}

async function renderClassBars(probabilitiesTensor) {
  const container = outputContainers.class;
  container.innerHTML = '';
  const probs = await probabilitiesTensor.data();
  const classNames = ['Sim', 'Não'];

  for (let i = 0; i < 2; i++) {
    const p = (probs[i] * 100).toFixed(1);
    const row = document.createElement('div');
    row.className = 'class-bar-row';
    row.innerHTML = `
      <div class="class-label" style="width: 220px; text-align: left;">${classNames[i]}</div>
      <div class="class-track">
        <div class="class-fill" style="width: ${p}%"></div>
      </div>
      <div class="class-prob">${p}%</div>
    `;
    container.appendChild(row);
  }
}

function renderMathExplanation(logits, probs) {
  const explanationBox = document.getElementById('softmaxExplanation');
  const mathSteps = document.getElementById('mathSteps');
  const mathConclusion = document.getElementById('mathConclusion');
  explanationBox.classList.remove('hidden');

  const classNames = ['Sim', 'Não'];
  const l1 = logits[0], l2 = logits[1];

  const formatNum = (v) => (Math.abs(v) > 9999 ? v.toExponential(2) : v.toFixed(4));
  const e1 = Math.exp(l1), e2 = Math.exp(l2);
  const sumE = e1 + e2;
  const p1 = (probs[0] * 100).toFixed(1);
  const p2 = (probs[1] * 100).toFixed(1);

  mathSteps.innerHTML = `
    <li>
      <strong>Passo 1 — Exponenciação dos Logits:</strong><br>
      A rede gerou os logits brutos <code>[${formatNum(l1)}, ${formatNum(l2)}]</code>.
      Aplica-se e<sup>x</sup> a cada um para amplificar a diferença:<br>
      ${classNames[0]}: e<sup>${formatNum(l1)}</sup> = ${formatNum(e1)}<br>
      ${classNames[1]}: e<sup>${formatNum(l2)}</sup> = ${formatNum(e2)}
    </li>
    <li>
      <strong>Passo 2 — Denominador comum (Σ):</strong><br>
      ${formatNum(e1)} + ${formatNum(e2)} = <strong>${formatNum(sumE)}</strong>
    </li>
    <li>
      <strong>Passo 3 — Probabilidade de cada classe:</strong><br>
      ${classNames[0]}: ${formatNum(e1)} ÷ ${formatNum(sumE)} = <strong>${p1}%</strong><br>
      ${classNames[1]}: ${formatNum(e2)} ÷ ${formatNum(sumE)} = <strong>${p2}%</strong>
    </li>
  `;

  const winnerIdx = probs[0] > probs[1] ? 0 : 1;
  const winnerProb = (probs[winnerIdx] * 100).toFixed(1);
  mathConclusion.innerHTML = `Os logits <code>[${formatNum(l1)}, ${formatNum(l2)}]</code> após o Softmax resultam em <strong>${winnerProb}%</strong> de confiança na classe <strong>${classNames[winnerIdx]}</strong>.`;
}
