// import React from 'react';
// import logo from './logo.svg';
// import './App.css';

// function App() {
//   return (
//     <div className="App">
//       <header className="App-header">
//         <img src={logo} className="App-logo" alt="logo" />
//         <p>
//           Edit <code>src/App.tsx</code> and save to reload.
//         </p>
//         <a
//           className="App-link"
//           href="https://reactjs.org"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           Learn React
//         </a>
//       </header>
//     </div>
//   );
// }

// export default App;

import React, { useState } from "react";
import axios, { AxiosResponse } from "axios";
import { PREDICT_ENDPOINT } from "./config";

interface InputData {
  asset: string;
  pred_datetime: string;
}

async function getPrediction(inputData: InputData): Promise<AxiosResponse> {
  try {
    const response = await axios.post(PREDICT_ENDPOINT, inputData);
    return response;
  } catch (error) {
    throw new Error(`An error occurred while connecting to the ML Predict Service Endpoint: ${error}`);
  }
}

function App() {
  const [inputData, setInputData] = useState<InputData>({
    asset: "",
    pred_datetime: "",
  });
  const [prediction, setPrediction] = useState<number>();

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    try {
      const response = await getPrediction(inputData);
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error(error);
    }
  }

  return (
    <div>
      <h1>Crypto Trading Predictions</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Asset:
          <select
            value={inputData.asset}
            onChange={(event) =>
              setInputData({ ...inputData, asset: event.target.value })
            }
          >
            <option value="">--Select--</option>
            <option value="btc-usd">BTC-USD</option>
            <option value="eth-usd">ETH-USD</option>
          </select>
        </label>
        <br />
        <label>
          Start Safe Entry Search - Date:
          <input
            type="date"
            value={inputData.pred_datetime}
            onChange={(event) =>
              setInputData({ ...inputData, pred_datetime: event.target.value })
            }
          />
        </label>
        <br />
        <button type="submit">Predict</button>
      </form>
      {prediction && <p>Prediction: {prediction}</p>}
    </div>
  );
}

export default App;