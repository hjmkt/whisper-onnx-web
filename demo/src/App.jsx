import { useState } from "react";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";
import audioUrl from "./assets/en.mp3";
import { Whisper, DecodingOptions } from "./whisper";
//import init from "@webonnx/wonnx-wasm";

let first = true;

async function convert() {
    if (!first) return;
    first = false;
    let whisper = new Whisper(true);
    let audio = await whisper.load_audio(audioUrl);
    audio = whisper.pad_or_trim(audio);
    let mel = await whisper.preprocessor(audio);
    let options = new DecodingOptions();
    let result = await whisper.decode(mel, options);
    return result;
}

function App() {
    const [count, setCount] = useState(0);
    convert();

    return (
        <div className="App">
            <div>
                <a href="https://vitejs.dev" target="_blank" rel="noreferrer">
                    <img src={viteLogo} className="logo" alt="Vite logo" />
                </a>
                <a href="https://reactjs.org" target="_blank" rel="noreferrer">
                    <img src={reactLogo} className="logo react" alt="React logo" />
                </a>
            </div>
            <h1>Vite + React</h1>
            <div className="card">
                <button
                    onClick={() => {
                        setCount((count) => count + 1);
                        loadAudio();
                        source.start(0);
                    }}
                >
                    count is {count}
                </button>
                <p>
                    Edit <code>src/App.jsx</code> and save to test HMR
                </p>
            </div>
            <p className="read-the-docs">Click on the Vite and React logos to learn more</p>
        </div>
    );
}

export default App;
