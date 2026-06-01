import * as tf from '@tensorflow/tfjs';

// ============================================================
// ARQUITETURA: Transfer Learning com MobileNet
// O MobileNet foi treinado no ImageNet (14 milhões de imagens).
// Usamos ele como "extrator de características" (features),
// congelamos seus pesos e treinamos a camada final com as fotos.
// Adicionalmente, usamos Similaridade de Cosseno no espaço de features (GAP)
// para rejeitar classes desconhecidas (fora do domínio).
// ============================================================

const IMAGE_SIZE = 224; // Tamanho que o MobileNet espera
let mobileNet = null;   // Rede pré-treinada (extrator de features)
let classifier = null;  // Nossa camada de classificação (treinável)

// Array de classes dinâmico, inicializado com 3 classes padrões
let classes = [
  { id: 'class_1', name: 'Moto', images: [], embeddings: [], gapVectors: [] },
  { id: 'class_2', name: 'Bicicleta', images: [], embeddings: [], gapVectors: [] },
  { id: 'class_3', name: 'Carro', images: [], embeddings: [], gapVectors: [] }
];

let testImageTensor = null;
let lastInferenceResults = null; // Guarda resultados para reavaliação instantânea com slider

// ---- Elementos da UI ----
const classesContainer = document.getElementById('classesContainer');
const addClassBtn = document.getElementById('addClassBtn');
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

const thresholdSlider = document.getElementById('thresholdSlider');
const thresholdVal = document.getElementById('thresholdVal');

const outputContainers = {
  conv: document.getElementById('convOutputs'),
  pool: document.getElementById('poolOutputs'),
  class: document.getElementById('classOutputs')
};

// ============================================================
// CARREGAMENTO DO MOBILENET
// ============================================================
async function loadMobileNet() {
  trainingStatus.textContent = 'Carregando MobileNet pré-treinado...';
  try {
    const mobilenet = await tf.loadLayersModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
    );
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    mobileNet = tf.model({ inputs: mobilenet.inputs, outputs: layer.output });

    trainingStatus.textContent = 'MobileNet carregado! Adicione imagens e inicie o treino.';
    trainingStatus.style.color = '#34d399';
    renderClasses();
  } catch (err) {
    trainingStatus.textContent = 'Erro ao carregar MobileNet: ' + err.message;
    trainingStatus.style.color = '#ef4444';
  }
}
loadMobileNet();

// ============================================================
// PRÉ-PROCESSAMENTO & CARACTERÍSTICAS
// ============================================================
function preprocessImage(img) {
  return tf.tidy(() => {
    return tf.browser.fromPixels(img)
      .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE])
      .toFloat()
      .div(127.5)
      .sub(1) // Normalização para [-1, 1]
      .expandDims(0);
  });
}

function extractFeatures(img) {
  return tf.tidy(() => {
    const preprocessed = preprocessImage(img);
    return mobileNet.predict(preprocessed); // [1, 7, 7, 256]
  });
}

// ============================================================
// RENDERIZADOR DINÂMICO DE CLASSES (DOM)
// ============================================================
function renderClasses() {
  classesContainer.innerHTML = '';

  classes.forEach((classItem, index) => {
    const card = document.createElement('div');
    card.className = 'didactic-card';
    card.dataset.id = classItem.id;

    // Header do card com input para mudar o nome
    const header = document.createElement('div');
    header.className = 'class-card-header';

    const inputName = document.createElement('input');
    inputName.type = 'text';
    inputName.className = 'class-name-input';
    inputName.value = classItem.name;
    inputName.addEventListener('change', (e) => {
      classItem.name = e.target.value;
      checkReadyToTrain();
    });

    header.appendChild(inputName);

    // Botão de remoção (apenas se tiver mais de 2 classes)
    if (classes.length > 2) {
      const removeBtn = document.createElement('button');
      removeBtn.className = 'btn-remove-class';
      removeBtn.innerHTML = '×';
      removeBtn.title = 'Remover esta classe';
      removeBtn.addEventListener('click', () => {
        // Liberar tensores
        classItem.embeddings.forEach(t => t.dispose());
        classItem.gapVectors.forEach(t => t.dispose());
        classes = classes.filter(c => c.id !== classItem.id);
        renderClasses();
        checkReadyToTrain();
      });
      header.appendChild(removeBtn);
    }

    card.appendChild(header);

    // Corpo do card
    const desc = document.createElement('p');
    desc.className = 'analogy';
    desc.style.marginBottom = '1rem';
    desc.textContent = `Envie imagens de exemplo da classe "${classItem.name}".`;
    card.appendChild(desc);

    // Upload de Imagens
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'image/*';
    fileInput.multiple = true;
    fileInput.style.display = 'none';

    const uploadBtn = document.createElement('button');
    uploadBtn.className = 'btn secondary';
    uploadBtn.textContent = 'Subir Imagens';
    uploadBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', async (e) => {
      const files = e.target.files;
      if (!files.length) return;

      uploadBtn.disabled = true;
      uploadBtn.textContent = 'Carregando...';

      for (let file of files) {
        const img = await loadImage(file);
        classItem.images.push(img);

        // Extrai características
        const embedding = extractFeatures(img);
        classItem.embeddings.push(embedding);

        // GAP Vector (Global Average Pooling) para similaridade de cosseno
        const gap = tf.tidy(() => tf.mean(embedding, [1, 2]).squeeze()); // [256]
        classItem.gapVectors.push(gap);
      }

      uploadBtn.disabled = false;
      uploadBtn.textContent = 'Subir Imagens';
      renderPreviews(classItem, card.querySelector('.image-preview-container'), card.querySelector('.count-badge'));
      checkReadyToTrain();
    });

    card.appendChild(fileInput);
    card.appendChild(uploadBtn);

    // Badge de quantidade
    const countBadge = document.createElement('div');
    countBadge.className = 'count-badge';
    countBadge.style.marginTop = '1rem';
    countBadge.textContent = `${classItem.images.length} imagens carregadas`;
    card.appendChild(countBadge);

    // Container de previews
    const previewContainer = document.createElement('div');
    previewContainer.className = 'image-preview-container';
    card.appendChild(previewContainer);

    classesContainer.appendChild(card);

    // Renderiza previews iniciais se já existirem imagens
    renderPreviews(classItem, previewContainer, countBadge);
  });
}

function renderPreviews(classItem, previewContainer, countBadge) {
  previewContainer.innerHTML = '';
  countBadge.textContent = `${classItem.images.length} imagens carregadas`;

  classItem.images.forEach((img, imgIdx) => {
    const wrap = document.createElement('div');
    wrap.className = 'image-preview-wrapper';

    const previewNode = document.createElement('img');
    previewNode.src = img.src;
    previewNode.className = 'image-preview';
    wrap.appendChild(previewNode);

    // Botão de deletar imagem individual
    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'btn-delete-img';
    deleteBtn.innerHTML = '×';
    deleteBtn.title = 'Remover esta foto';
    deleteBtn.addEventListener('click', () => {
      // Liberar memória dos tensores individuais
      classItem.embeddings[imgIdx].dispose();
      classItem.gapVectors[imgIdx].dispose();

      // Remover dos arrays
      classItem.images.splice(imgIdx, 1);
      classItem.embeddings.splice(imgIdx, 1);
      classItem.gapVectors.splice(imgIdx, 1);

      // Re-renderizar
      renderPreviews(classItem, previewContainer, countBadge);
      checkReadyToTrain();
    });

    wrap.appendChild(deleteBtn);
    previewContainer.appendChild(wrap);
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

// Botão Adicionar Classe
addClassBtn.addEventListener('click', () => {
  if (classes.length >= 6) {
    alert('Você atingiu o limite de 6 classes para garantir boa performance.');
    return;
  }
  const nextNum = classes.length + 1;
  classes.push({
    id: `class_${Date.now()}`,
    name: `Classe ${nextNum}`,
    images: [],
    embeddings: [],
    gapVectors: []
  });
  renderClasses();
  checkReadyToTrain();
});

function checkReadyToTrain() {
  const allHaveImages = classes.every(c => c.images.length > 0);
  const minClasses = classes.length >= 2;

  if (allHaveImages && minClasses) {
    trainBtn.disabled = false;
    trainingStatus.textContent = `Pronto! Envie pelo menos 1 imagem por classe e inicie o treinamento.`;
    trainingStatus.style.color = '#34d399';
  } else {
    trainBtn.disabled = true;
    trainingStatus.textContent = `Aguardando imagens... Envie fotos para todas as classes cadastradas.`;
    trainingStatus.style.color = '#cbd5e1';
  }
}

// ============================================================
// CONTROLE DO SLIDER
// ============================================================
thresholdSlider.addEventListener('input', (e) => {
  thresholdVal.textContent = `${e.target.value}%`;
  // Se já tivermos rodado uma inferência, reavalia o resultado instantaneamente!
  if (lastInferenceResults) {
    evaluateInference(lastInferenceResults.logits, lastInferenceResults.probs, lastInferenceResults.maxSimilarity, lastInferenceResults.closestClass);
  }
});

// ============================================================
// TREINAMENTO MULTICLASSE (TRANSFER LEARNING)
// ============================================================
trainBtn.addEventListener('click', async () => {
  trainBtn.disabled = true;
  addClassBtn.disabled = true;
  // Desabilitar botões de upload das classes
  document.querySelectorAll('.didactic-card button.btn.secondary').forEach(b => b.disabled = true);

  trainingStatus.textContent = 'Construindo classificador multiclasse...';
  trainingStatus.style.color = '#f8fafc';

  const sampleShape = classes[0].embeddings[0].shape.slice(1); // ex: [7, 7, 256]

  // Modelo Sequencial MLP
  classifier = tf.sequential();
  classifier.add(tf.layers.flatten({ inputShape: sampleShape }));
  classifier.add(tf.layers.dense({ units: 64, activation: 'relu', name: 'hidden_1' }));
  classifier.add(tf.layers.dropout({ rate: 0.3 }));
  classifier.add(tf.layers.dense({ units: classes.length, name: 'dense_out' })); // C saídas
  classifier.add(tf.layers.softmax({ name: 'softmax_out' }));

  classifier.compile({
    optimizer: tf.train.adam(0.0005),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  // Montar dataset (Embeddings + One-Hot Labels)
  const allEmbeddings = [];
  const labels = [];

  classes.forEach((classItem, classIdx) => {
    classItem.embeddings.forEach(emb => {
      allEmbeddings.push(emb);
      
      // Cria vetor One-Hot
      const oneHot = Array(classes.length).fill(0);
      oneHot[classIdx] = 1;
      labels.push(oneHot);
    });
  });

  const xs = tf.tidy(() => tf.concat(allEmbeddings, 0)); // [N, 7, 7, 256]
  const ys = tf.tensor2d(labels, [labels.length, classes.length]);

  trainProgress.classList.remove('hidden');
  lossText.classList.remove('hidden');
  trainingStatus.textContent = 'Treinando rede densa sobre características convolucionais...';

  const epochs = 100;
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
  
  // Reabilitar controles
  addClassBtn.disabled = false;
  document.querySelectorAll('.didactic-card button.btn.secondary').forEach(b => b.disabled = false);
  btnTest.disabled = false;
});

// ============================================================
// TESTES E INFERÊNCIA
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
  testImageTensor = preprocessImage(img);
  testImageTensor._rawImg = img;

  runInferenceBtn.disabled = false;
  networkVisualizer.classList.add('hidden');
  lastInferenceResults = null;
});

// Função de Cosseno Math
function cosineSimilarity(a, b) {
  return tf.tidy(() => {
    const aNorm = tf.div(a, tf.norm(a).add(1e-5));
    const bNorm = tf.div(b, tf.norm(b).add(1e-5));
    return tf.sum(tf.mul(aNorm, bNorm)).arraySync();
  });
}

runInferenceBtn.addEventListener('click', async () => {
  if (!classifier || !testImageTensor) return;
  runInferenceBtn.disabled = true;
  networkVisualizer.classList.remove('hidden');

  // 1. Extrair mapa de features via MobileNet
  const testEmbedding = mobileNet.predict(testImageTensor); // [1, 7, 7, 256]

  // 2. Extrair vetor GAP para similaridade de cosseno
  const testGap = tf.tidy(() => tf.mean(testEmbedding, [1, 2]).squeeze()); // [256]

  // 3. Achar a similaridade máxima com os exemplos de treino
  let maxSimilarity = -1;
  let closestClass = null;

  classes.forEach(c => {
    c.gapVectors.forEach(trainGap => {
      const sim = cosineSimilarity(testGap, trainGap);
      if (sim > maxSimilarity) {
        maxSimilarity = sim;
        closestClass = c;
      }
    });
  });

  // 4. Executar inferência na rede MLP
  const denseOutLayer = classifier.getLayer('dense_out');
  const logitsModel = tf.model({ inputs: classifier.inputs, outputs: denseOutLayer.output });

  const logitsTensor = logitsModel.predict(testEmbedding);
  const finalOut = classifier.predict(testEmbedding);

  const logitsData = await logitsTensor.data();
  const probsData = await finalOut.data();

  // Guardar resultados para reavaliações dinâmicas do slider de sensibilidade
  lastInferenceResults = {
    logits: Array.from(logitsData),
    probs: Array.from(probsData),
    maxSimilarity,
    closestClass
  };

  // 5. Renderizar os mapas de ativação
  await renderActivationMaps(testEmbedding, outputContainers.conv, 4);

  // 6. MaxPooling visual simulado
  const pooled = tf.tidy(() => tf.maxPool(testEmbedding, 2, 2, 'valid'));
  await renderActivationMaps(pooled, outputContainers.pool, 4);

  // 7. Avaliar a classificação final baseada no slider e nos logits
  evaluateInference(Array.from(logitsData), Array.from(probsData), maxSimilarity, closestClass);

  // Limpar tensores
  testGap.dispose();
  pooled.dispose();
  tf.dispose([testEmbedding, logitsTensor, finalOut]);
  runInferenceBtn.disabled = false;
});

// ============================================================
// DECISÃO DE INFERÊNCIA & APRESENTAÇÃO
// ============================================================
function evaluateInference(logits, probs, maxSimilarity, closestClass) {
  const threshold = parseFloat(thresholdSlider.value);
  const similarityPercent = maxSimilarity * 100;
  
  const container = outputContainers.class;
  container.innerHTML = '';

  const explanationBox = document.getElementById('softmaxExplanation');
  const mathSteps = document.getElementById('mathSteps');
  const mathConclusion = document.getElementById('mathConclusion');
  explanationBox.classList.remove('hidden');

  const winnerIdx = probs.indexOf(Math.max(...probs));
  const isRecognized = similarityPercent >= threshold;

  // Renderizar barras de probabilidade
  classes.forEach((c, i) => {
    const p = (probs[i] * 100).toFixed(1);
    const isWinner = i === winnerIdx;
    const barColor = isRecognized 
      ? 'linear-gradient(90deg, var(--accent-secondary), var(--accent-primary))' 
      : 'linear-gradient(90deg, #64748b, #475569)';

    const row = document.createElement('div');
    row.className = 'class-bar-row';
    row.innerHTML = `
      <div class="class-label" style="width: 200px; text-align: left; color: ${isWinner && isRecognized ? '#34d399' : 'inherit'}">
        ${c.name} ${isWinner && isRecognized ? ' 🎯' : ''}
      </div>
      <div class="class-track">
        <div class="class-fill" style="width: ${p}%; background: ${barColor}"></div>
      </div>
      <div class="class-prob">${p}%</div>
    `;
    container.appendChild(row);
  });

  // Mostra bloco extra de "Reconhecido" vs "Desconhecido" no resultado
  const resultAlert = document.createElement('div');
  resultAlert.style.marginTop = '1.5rem';
  resultAlert.style.padding = '1rem';
  resultAlert.style.borderRadius = '6px';
  resultAlert.style.textAlign = 'center';
  resultAlert.style.fontWeight = 'bold';
  resultAlert.style.fontSize = '1.1rem';

  if (isRecognized) {
    resultAlert.style.background = 'rgba(52, 211, 153, 0.15)';
    resultAlert.style.color = '#34d399';
    resultAlert.style.border = '1px solid rgba(52, 211, 153, 0.3)';
    resultAlert.innerHTML = `Objeto Identificado: <strong>${classes[winnerIdx].name}</strong>! 🎉<br>
      <span style="font-size: 0.85rem; font-weight: normal; color: var(--text-secondary);">
        Similaridade visual de ${similarityPercent.toFixed(1)}% (limiar mínimo: ${threshold}%)
      </span>`;
  } else {
    resultAlert.style.background = 'rgba(239, 68, 68, 0.15)';
    resultAlert.style.color = '#f87171';
    resultAlert.style.border = '1px solid rgba(239, 68, 68, 0.3)';
    resultAlert.innerHTML = `Desconhecido / Não identificado 🚫<br>
      <span style="font-size: 0.85rem; font-weight: normal; color: var(--text-secondary);">
        A similaridade visual de ${similarityPercent.toFixed(1)}% com a classe mais próxima (${closestClass ? closestClass.name : 'Nenhuma'}) ficou abaixo do limiar de detecção (${threshold}%)
      </span>`;
  }
  container.appendChild(resultAlert);

  // Renderizar explicação da matemática Softmax Dinâmica para C classes
  const formatNum = (v) => (Math.abs(v) > 9999 ? v.toExponential(2) : v.toFixed(4));
  const exps = logits.map(l => Math.exp(l));
  const sumExp = exps.reduce((acc, curr) => acc + curr, 0);

  let stepsHTML = `
    <li>
      <strong>Passo 1 — Exponenciação de Logits (e<sup>x</sup>):</strong><br>
      A rede densa produziu as pontuações brutas (logits): <code>[${logits.map(formatNum).join(', ')}]</code>.<br>
      Aplica-se o número de Euler <em>e</em> para amplificar as diferenças e remover valores negativos:<br>
      ${classes.map((c, i) => `&bull; ${c.name}: e<sup>${formatNum(logits[i])}</sup> = <strong>${formatNum(exps[i])}</strong>`).join('<br>')}
    </li>
    <li>
      <strong>Passo 2 — Denominador Comum (Σ e<sup>x</sup>):</strong><br>
      Soma de todos os valores exponenciados:<br>
      ${exps.map(formatNum).join(' + ')} = <strong>${formatNum(sumExp)}</strong>
    </li>
    <li>
      <strong>Passo 3 — Probabilidade Softmax (e<sup>x</sup> / Σ):</strong><br>
      Divide-se cada termo individual pela soma para normalizar a probabilidade entre 0% e 100%:<br>
      ${classes.map((c, i) => `&bull; ${c.name}: ${formatNum(exps[i])} ÷ ${formatNum(sumExp)} = <strong>${(probs[i] * 100).toFixed(1)}%</strong>`).join('<br>')}
    </li>
    <li>
      <strong>Passo 4 — Similaridade Convolucional (Out-of-Distribution):</strong><br>
      Similaridade de Cosseno entre a imagem e o exemplo cadastrado de <strong>${closestClass ? closestClass.name : 'nenhum'}</strong>: <strong>${similarityPercent.toFixed(1)}%</strong>.<br>
      Decisão do Limiar (${threshold}%): <strong>${isRecognized ? 'ACEITO' : 'REJEITADO (Marcado como Desconhecido)'}</strong>.
    </li>
  `;
  mathSteps.innerHTML = stepsHTML;

  if (isRecognized) {
    mathConclusion.innerHTML = `O classificador CNN escolheu <strong>${classes[winnerIdx].name}</strong> com <strong>${(probs[winnerIdx] * 100).toFixed(1)}%</strong> de probabilidade, e a similaridade visual foi validada com sucesso!`;
  } else {
    mathConclusion.innerHTML = `Embora a CNN aponte matematicamente para <strong>${classes[winnerIdx].name}</strong>, a similaridade de cosseno (${similarityPercent.toFixed(1)}%) acusa que esta imagem é de um objeto desconhecido (fora do domínio de treino).`;
  }
}

// ============================================================
// RENDERIZADOR DE ATIVAÇÕES CONV2D & MAXPOOLING
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
