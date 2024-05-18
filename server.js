const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const { createCanvas, loadImage } = require('canvas');
const fs = require('fs');

const app = express();
const port = 3000;

// Configuração do Multer para armazenar os arquivos em 'uploads/'
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + '-' + file.originalname);
    }
});

const upload = multer({ storage: storage });

// Função assíncrona para carregar o modelo
async function loadModel() {
const model = await tf.loadLayersModel('file:///home/mohamed/node-api/modelo_gatos_cachorros.h5');
    return model;
}

// Rota para fazer previsões
app.post('/predict', upload.single('image'), async (req, res) => {
    try {
        const image = await loadImage(req.file.path);
        const canvas = createCanvas(150, 150);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0, 150, 150);

        const imgData = ctx.getImageData(0, 0, 150, 150);
        const imgArray = new Float32Array(imgData.data).map(v => v / 255);

        const input = tf.tensor([imgArray]);
        const model = await loadModel();
        const prediction = model.predict(input);

        const result = prediction.dataSync()[0] > 0.5 ? 'Cachorro' : 'Gato';

        res.json({ result });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Erro ao fazer previsão' });
    }
});

// Iniciar o servidor
app.listen(port, async () => {
    console.log(`Servidor rodando em http://localhost:${port}`);
    await loadModel(); // Carregar o modelo quando o servidor iniciar
});
