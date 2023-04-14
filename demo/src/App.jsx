import { useState } from "react";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";
import audioUrl from "./assets/en.mp3";
import { Whisper, DecodingOptions } from "@core/whisper";
import encoderWasmModelUrl from "@models/encoder_int8.onnx";
import decoderWasmModelUrl from "@models/decoder_int8.onnx";
import preprocessorWasmModelUrl from "@models/preprocessor_int8.onnx";
import positionalEmbeddingUrl from "@models/positional_embedding.bin.gz";

const CHUNK_LENGTH = 10;

async function convert() {
    let whisper = new Whisper(
        preprocessorWasmModelUrl,
        encoderWasmModelUrl,
        decoderWasmModelUrl,
        positionalEmbeddingUrl,
        CHUNK_LENGTH,
        true
    );
    let audio = await whisper.load_audio(audioUrl);
    audio = whisper.pad_or_trim(audio);
    let mel = await whisper.preprocessor(audio);
    let options = new DecodingOptions();
    let result = await whisper.decode(mel, options);
    return result;
}

function App() {
    const [count, setCount] = useState(0);

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
