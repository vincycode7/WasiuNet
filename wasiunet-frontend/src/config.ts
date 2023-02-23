// # Load secrets

const ML_BASE_URL = process.env.ML_BASE_URL ? process.env.ML_BASE_URL : ""
const ML_PORT = process.env.ML_PORT ? process.env.ML_PORT : ""
const PREDICT_ENDPOINT = ML_BASE_URL+":"+ML_PORT.toString()+"/predict"

export {PREDICT_ENDPOINT}