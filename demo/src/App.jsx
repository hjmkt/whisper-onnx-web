import { useState, useEffect } from "react";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";
import audioUrl from "./assets/en.mp3";
import { Whisper, DecodingOptions, Tensor } from "@core/whisper";
import positionalEmbeddingBaseUrl from "@models/positional_embedding_base.bin.gz";
import positionalEmbeddingSmallUrl from "@models/positional_embedding_small.bin.gz";
import processorCode from "./whisper_processor.js?raw";

const CHUNK_LENGTH = 10;
let model_type = "small";
import preprocessorBaseModelUrl from "@models/preprocessor_base_5s_int8.onnx.gz";
import preprocessorSmallModelUrl from "@models/preprocessor_small_10s_int8.onnx.gz";
import encoderBaseModelUrl from "@models/encoder_base_5s_int8.onnx.gz";
import encoderSmallModelUrl from "@models/encoder_small_10s_int8.onnx.gz";
import decoderBaseModelUrl from "@models/decoder_base_5s_int8.onnx.gz";
import decoderSmallModelUrl0 from "@models/decoder_small_10s_int8.onnx.0.gz";
import decoderSmallModelUrl1 from "@models/decoder_small_10s_int8.onnx.1.gz";
import WhisperWorker from "./whisper_worker?worker";

let whisper_workers = {
    base: {
        0: { running: false, worker: new WhisperWorker() },
        1: { running: false, worker: new WhisperWorker() },
    },
    small: {
        0: { running: false, worker: new WhisperWorker() },
        1: { running: false, worker: new WhisperWorker() },
    },
};

let whisperInitialized = false;
let lastBaseTexts = [];
let lastSmallText = "";
let tmpSmallText = "";
let fixedText = "";
let chunkIndex = 0;

async function initWhisper(textCallback = () => {}) {
    if (!whisperInitialized) {
        Promise.all([
            fetch(positionalEmbeddingBaseUrl).then((r) => r.arrayBuffer()),
            fetch(preprocessorBaseModelUrl).then((r) => r.arrayBuffer()),
            fetch(encoderBaseModelUrl).then((r) => r.arrayBuffer()),
            fetch(decoderBaseModelUrl).then((r) => r.arrayBuffer()),
        ]).then((values) => {
            let [
                positionalEmbeddingBase,
                preprocessorBase,
                encoderBase,
                decoderBase,
            ] = values;

            console.log(
                "init post",
                positionalEmbeddingBase,
                preprocessorBase,
                encoderBase,
                decoderBase
            );
            for (let i = 0; i < 2; i++) {
                whisper_workers["base"][i].worker.postMessage({
                    type: "init",
                    modelType: "base",
                    positionalEmbedding: positionalEmbeddingBase,
                    preprocessorModel: preprocessorBase,
                    encoderModel: encoderBase,
                    decoderModel: decoderBase,
                    chunkLength: 5,
                    debug: false,
                });
            }
        });
        Promise.all([
            fetch(positionalEmbeddingSmallUrl).then((r) => r.arrayBuffer()),
            fetch(preprocessorSmallModelUrl).then((r) => r.arrayBuffer()),
            fetch(encoderSmallModelUrl).then((r) => r.arrayBuffer()),
            fetch(decoderSmallModelUrl0).then((r) => r.arrayBuffer()),
            fetch(decoderSmallModelUrl1).then((r) => r.arrayBuffer()),
        ]).then((values) => {
            let [
                positionalEmbeddingSmall,
                preprocessorSmall,
                encoderSmall,
                decoderSmall0,
                decoderSmall1,
            ] = values;
            let n_bytes = decoderSmall0.byteLength + decoderSmall1.byteLength;
            let buffer = new ArrayBuffer(n_bytes);
            buffer = new Uint8Array(buffer);
            buffer.set(new Uint8Array(decoderSmall0), 0);
            buffer.set(new Uint8Array(decoderSmall1), decoderSmall0.byteLength);
            let decoderSmall = buffer.buffer;
            for (let i = 0; i < 2; i++) {
                whisper_workers["small"][i].worker.postMessage({
                    type: "init",
                    modelType: "small",
                    positionalEmbedding: positionalEmbeddingSmall,
                    preprocessorModel: preprocessorSmall,
                    encoderModel: encoderSmall,
                    decoderModel: decoderSmall,
                    chunkLength: 10,
                    debug: false,
                });
            }
        });
    }
    for (let i = 0; i < 2; i++) {
        whisper_workers["base"][i].worker.onmessage = (e) => {
            console.log("base", e.data);
            if (e.data.action == "result") {
                lastBaseTexts.push(e.data.payload);
                let headText = lastBaseTexts.slice(0, 2).join("");
                let tailText = lastBaseTexts.slice(2).join("");
                if (tmpSmallText.length >= headText.length) {
                    headText = tmpSmallText;
                } else {
                    headText =
                        tmpSmallText + headText.slice(tmpSmallText.length);
                }
                let text = fixedText + headText + tailText;
                ((p) => textCallback(p))(text);
                whisper_workers["base"][i].running = false;
            } else {
                let tmpTexts = [];
                for (let i = 0; i < lastBaseTexts.length; i++) {
                    tmpTexts.push(lastBaseTexts[i]);
                }
                tmpTexts.push(e.data.payload);
                let headText = tmpTexts.slice(0, 2).join("");
                let tailText = tmpTexts.slice(2).join("");
                if (tmpSmallText.length >= headText.length) {
                    headText = tmpSmallText;
                } else {
                    headText =
                        tmpSmallText + headText.slice(tmpSmallText.length);
                }
                let text = fixedText + headText + tailText;
                ((p) => textCallback(p))(text);
                whisper_workers["base"][i].running = false;
            }
        };
    }
    for (let i = 0; i < 2; i++) {
        whisper_workers["small"][i].worker.onmessage = (e) => {
            console.log("small", e.data);
            if (e.data.action == "result") {
                if (lastBaseTexts.length >= 2) {
                    lastBaseTexts = lastBaseTexts.slice(2);
                }
                lastSmallText = e.data.payload;
                fixedText += lastSmallText;
                tmpSmallText = "";
                let text = fixedText + lastBaseTexts.join("");
                ((p) => textCallback(p))(text);
                whisper_workers["small"][i].running = false;
            } else {
                tmpSmallText = e.data.payload;
                let headText = lastBaseTexts.slice(0, 2).join("");
                let tailText = lastBaseTexts.slice(2).join("");
                if (tmpSmallText.length >= headText.length) {
                    headText = tmpSmallText;
                } else {
                    headText =
                        tmpSmallText + headText.slice(tmpSmallText.length);
                }
                let text = fixedText + headText + tailText;
                ((p) => textCallback(p))(text);
            }
        };
    }
    whisperInitialized = true;
}

async function getWhisper(model_type) {
    let workers = whisper_workers[model_type];
    let worker = null;
    while (worker == null) {
        for (let i = 0; i < 2; i++) {
            if (!workers[i].running) {
                worker = workers[i];
                break;
            }
        }
        if (worker == null) {
            await new Promise((resolve) => setTimeout(resolve, 100));
        }
    }
    worker.running = true;
    return worker;
}

function runWhisper(
    model_type,
    tensor,
    textCallback,
    tokenCallback = () => {}
) {
    console.log("runWhisper");
    getWhisper(model_type).then((worker) => {
        worker.worker.postMessage({
            type: "run",
            tensor: {
                shape: tensor.shape,
                data: tensor.data,
                dtype: tensor.dtype,
            },
        });
    });
}

async function convert() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: false,
        audio: true,
    });
    const tmpCtx = new AudioContext({ sampleRate: 16000 });
    const source = tmpCtx.createMediaStreamSource(stream);
    const processorBlob = new Blob([processorCode], {
        type: "text/javascript",
    });
    const processorUrl = URL.createObjectURL(processorBlob);
    await tmpCtx.audioWorklet.addModule(processorUrl).then(() => {
        const node = new AudioWorkletNode(tmpCtx, "whisper-processor", {
            processorOptions: {
                chunkLength: CHUNK_LENGTH,
            },
        });
        node.port.onmessage = (e) => {
            let buffer = new Float32Array(e.data);
            let tensor = new Tensor([buffer.length], "float32", buffer);
            runWhisper("base", tensor);
            if (chunkIndex % 2 == 1) {
                runWhisper("small", tensor);
            }
            chunkIndex++;
        };
        source.connect(node);
    });
}

function App() {
    const [count, setCount] = useState(0);
    const [text, setText] = useState("");
    let texts = [""];

    const updateText = (text) => {
        texts[0] = text;
        //console.log("texts", texts[0]);
    };
    useEffect(() => {
        initWhisper(setText);
    });

    //useEffect(() => {
    //const interval = setInterval(() => {
    ////texts[0] = texts[0] + "a";
    //setText(texts[0]);
    //console.log("texts", texts[0]);
    //}, 100);
    //return () => clearInterval(interval);
    //}, []);

    return (
        <div className="App">
            <div>
                <a href="https://vitejs.dev" target="_blank" rel="noreferrer">
                    <img src={viteLogo} className="logo" alt="Vite logo" />
                </a>
                <a href="https://reactjs.org" target="_blank" rel="noreferrer">
                    <img
                        src={reactLogo}
                        className="logo react"
                        alt="React logo"
                    />
                </a>
            </div>
            <h1>Vite + React</h1>
            <textarea
                readOnly
                className="textarea"
                rows="6"
                cols="150"
                onChange={(e) => setText(e.target.value)}
                value={text}
            />
            <div className="card">
                <button
                    onClick={() => {
                        setCount((count) => count + 1);
                        convert();
                        //source.start(0);
                    }}
                >
                    count is {count}
                </button>
                <p>
                    Edit <code>src/App.jsx</code> and save to test HMR
                </p>
            </div>
            <p className="read-the-docs">
                Click on the Vite and React logos to learn more
            </p>
        </div>
    );
}

export default App;
