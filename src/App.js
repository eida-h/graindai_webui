import React, { useState, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import './App.css';

function App() {
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const imageRef = useRef(null);
  const labels = ['coarse', 'fine', 'medium', 'loading'];

  useEffect(() => {
    // モデルをロードする関数を定義
    const loadModel = async () => {
      const loadedModel = await tf.loadLayersModel(process.env.PUBLIC_URL + '/model/model.json');
      setModel(loadedModel);
    };

    // 定義した関数を実行
    loadModel();
  }, []); // 空の依存配列を渡して、このuseEffectを一度だけ実行する


  const predictImage = () => {
    const image = tf.browser.fromPixels(imageRef.current)
      .resizeNearestNeighbor([224, 224])
      .toFloat()
      .expandDims();
    const predictions = model.predict(image);
    const predictedClass = predictions.argMax(1).dataSync()[0];
    setPrediction(predictedClass);  // Here, you'd probably have some mapping from class index to label
  };

  return (
    <div className="App">
      <h1>Coffee beans grind predictor</h1>
      <input type="file" onChange={e => {
        if (!e.target.files || e.target.files.length === 0) {
          return;  // ファイルが選択されていない場合は何もしない
        }
        const file = e.target.files[0];
        const reader = new FileReader();
        reader.onload = () => {
          imageRef.current.src = reader.result;
        };
        reader.readAsDataURL(file);
      }} />
      <img ref={imageRef} alt="To be predicted" width="224" height="224"/>
      <button onClick={predictImage}>Predict</button>
      {prediction !== null && <div>Prediction: {labels[prediction]}</div>}
    </div>
  );
}

export default App;
