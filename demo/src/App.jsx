import { useState, useEffect } from "react";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";
import audioUrl from "./assets/en.mp3";
import { Whisper, DecodingOptions, Tensor } from "@core/whisper";
import positionalEmbeddingBaseUrl from "@models/positional_embedding_base.bin.compressed";
import positionalEmbeddingSmallUrl from "@models/positional_embedding_small.bin.compressed";
import processorCode from "./whisper_processor.js?raw";

const CHUNK_LENGTH = 10;
import preprocessorBaseModelUrl from "@models/preprocessor_base_5s_int8.onnx.compressed";
import preprocessorSmallModelUrl from "@models/preprocessor_small_10s_int8.onnx.compressed";
import encoderBaseModelUrl from "@models/encoder_base_5s_int8.onnx.compressed";
import encoderSmallModelUrl from "@models/encoder_small_10s_int8.onnx.compressed";
import decoderBaseModelUrl from "@models/decoder_base_5s_int8.onnx.compressed";
import decoderSmallModelUrl0 from "@models/decoder_small_10s_int8.onnx.0.compressed";
import decoderSmallModelUrl1 from "@models/decoder_small_10s_int8.onnx.1.compressed";
import WhisperWorker from "./whisper_worker?worker";
import Circle from "react-circle";

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
let tmpSmallText = "";
let fixedText = "";
let chunkIndex = 0;

async function initWhisper(
    progressCallback,
    textCallback = () => {},
    isProduction
) {
    if (!whisperInitialized) {
        Promise.all([
            fetch(positionalEmbeddingBaseUrl),
            fetch(preprocessorBaseModelUrl),
            fetch(encoderBaseModelUrl),
            fetch(decoderBaseModelUrl),
            fetch(positionalEmbeddingSmallUrl),
            fetch(preprocessorSmallModelUrl),
            fetch(encoderSmallModelUrl),
            fetch(decoderSmallModelUrl0),
            fetch(decoderSmallModelUrl1),
        ]).then((fetchRes) => {
            const lengths = fetchRes.map((v) =>
                Number(v.headers.get("content-length"))
            );
            const total = lengths.reduce((a, b) => a + b, 0);
            let buffers = lengths.map((v) => new Uint8Array(v));
            let readers = fetchRes.map((v) => v.body.getReader());
            let chunks = [0, 0, 0, 0, 0, 0, 0, 0, 0];
            let chunk = 0;
            Promise.all(readers.map((v) => v.read())).then(function process(
                res
            ) {
                if (res.every((v) => v.done)) {
                    Promise.all(
                        buffers.map((v) =>
                            new Response(
                                new Blob([v])
                                    .stream()
                                    .pipeThrough(
                                        new DecompressionStream("gzip")
                                    )
                            ).arrayBuffer()
                        )
                    ).then((v) => {
                        for (let i = 0; i < 2; i++) {
                            whisper_workers["base"][i].worker.postMessage({
                                type: "init",
                                modelType: "base",
                                positionalEmbedding: v[0],
                                preprocessorModel: v[1],
                                encoderModel: v[2],
                                decoderModel: v[3],
                                chunkLength: 5,
                                debug: false,
                                isProduction: isProduction,
                            });
                        }
                        let n_bytes = v[7].byteLength + v[8].byteLength;
                        let buffer = new ArrayBuffer(n_bytes);
                        buffer = new Uint8Array(buffer);
                        buffer.set(new Uint8Array(v[7]), 0);
                        buffer.set(new Uint8Array(v[8]), v[7].byteLength);
                        let decoderSmall = buffer.buffer;
                        for (let i = 0; i < 2; i++) {
                            whisper_workers["small"][i].worker.postMessage({
                                type: "init",
                                modelType: "small",
                                positionalEmbedding: v[4],
                                preprocessorModel: v[5],
                                encoderModel: v[6],
                                decoderModel: decoderSmall,
                                chunkLength: 10,
                                debug: false,
                                isProduction: isProduction,
                            });
                        }
                        progressCallback(100);
                    });
                    return;
                }
                for (let i = 0; i < res.length; i++) {
                    if (!res[i].done) {
                        if (
                            buffers[i].byteLength <
                            chunks[i] + res[i].value.byteLength
                        ) {
                            let newBuffer = new Uint8Array(
                                chunks[i] + res[i].value.byteLength
                            );
                            newBuffer.set(buffers[i], 0);
                            buffers[i] = newBuffer;
                        }
                        buffers[i].set(res[i].value, chunks[i]);
                        chunks[i] += res[i].value.byteLength;
                        chunk += res[i].value.byteLength;
                    }
                }
                progressCallback(Math.floor((chunk / total) * 99));
                Promise.all(readers.map((v) => v.read())).then(process);
            });
        });
    }
    for (let i = 0; i < 2; i++) {
        whisper_workers["base"][i].worker.onmessage = (e) => {
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
            if (e.data.action == "result") {
                if (lastBaseTexts.length >= 2) {
                    lastBaseTexts = lastBaseTexts.slice(2);
                }
                fixedText += e.data.payload;
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

async function transcribe() {
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

const App = ({ isProduction }) => {
    const [count, setCount] = useState(0);
    const [progress, setProgress] = useState(0);
    const [text, setText] = useState("");

    useEffect(() => {
        initWhisper(setProgress, setText, isProduction);
    });

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
            <div>
                <Circle progress={progress} />
            </div>
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
                        transcribe();
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
};

export default App;
