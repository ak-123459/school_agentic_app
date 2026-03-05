import subprocess
import os
import json
import uuid
import tempfile
import time
import requests

class PyPiper():
    def __init__(self):
        t0 = time.time()
        print(f"[init] Starting PyPiper init...")

        # ✅ Always relative to THIS file, not the run directory
        self.BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
        self.voices_dir = os.path.join(self.BASE_DIR, "..", "voices")  # app/voices
        self.voices_dir = os.path.normpath(self.voices_dir)           # clean up ../

        os.makedirs(self.voices_dir, exist_ok=True)

        voices_json = os.path.join(self.voices_dir, "voices.json")

        if not os.path.isfile(voices_json):
            print(f"[init] voices.json not found, downloading...")
            t1 = time.time()
            voice_file = requests.get("https://huggingface.co/rhasspy/piper-voices/raw/main/voices.json")
            with open(voices_json, 'wb') as w:
                w.write(voice_file.content)
            print(f"[init] voices.json downloaded in {time.time()-t1:.2f}s")

        with open(voices_json, "rb") as file:
            voice_main = json.loads(file.read())

        self.key_list = list(voice_main.keys())
        self.model    = "en_US-bryce-medium"
        self.piper_exe = r"C:\piper\piper.exe"
        print(f"[init] Done in {time.time()-t0:.2f}s — {len(self.key_list)} voices available")

    def load_mod(self, instr="en_US-bryce-medium", model_dir=None):
        t0 = time.time()
        self.model = instr
        lang  = instr.split("_")[0]
        dia   = instr.split("-")[0]
        name  = instr.split("-")[1]
        style = instr.split("-")[2]
        file  = f'{instr}.onnx'

        print(f"[load_mod] Requested model: {file}")

        # ✅ Use passed dir, else default to app/voices/
        if model_dir is None:
            model_dir = self.voices_dir

        model_path = os.path.join(model_dir, file)
        json_path  = os.path.join(model_dir, f'{file}.json')

        if not os.path.isfile(model_path):
            print(f"[load_mod] Not found locally, downloading...")
            m_path = (
                f"https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                f"{lang}/{dia}/{name}/{style}/{file}"
            )
            json_file = requests.get(f"{m_path}.json")
            mod_file  = requests.get(m_path)

            os.makedirs(model_dir, exist_ok=True)
            with open(model_path, 'wb') as m:
                m.write(mod_file.content)
            with open(json_path, 'wb') as j:
                j.write(json_file.content)
        else:
            print(f"[load_mod] Found locally: {model_path}")

        self.model_path = model_path
        self.json_ob    = json_path
        print(f"[load_mod] Ready in {time.time()-t0:.2f}s")


    def _get_sample_rate(self, model):
        json_path = os.path.join(os.getcwd(), 'voices', f'{model}.onnx.json')
        with open(json_path, 'r') as f:
            config = json.load(f)
        return config['audio']['sample_rate']

    def tts(self, in_text, model="", length=2, noise=0.1, width=1, sen_pause=1):
        t0 = time.time()
        if not model:
            model = self.model

        self.load_mod(instr=model)

        text = in_text.replace(". ", ".\n")
        model_path = os.path.join(os.getcwd(), 'voices', f'{model}.onnx')
        json_path = os.path.join(os.getcwd(), 'voices', f'{model}.onnx.json')
        output_file = f"{uuid.uuid4()}.wav"

        print(f"[tts] Writing temp input file...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False, encoding='utf-8') as tmp:
            tmp.write(text)
            tmp_path = tmp.name
        print(f"[tts] Temp file: {tmp_path}")

        try:
            print(f"[tts] Launching piper...")
            t1 = time.time()
            process = subprocess.Popen(
                [
                    self.piper_exe,
                    "--model", model_path,
                    "--config", json_path,
                    "--output_file", output_file,
                    "--length_scale", str(length),
                    "--noise_scale", str(noise),
                    "--noise_w", str(width),
                    "--sentence_silence", str(sen_pause)
                ],
                stdin=open(tmp_path, 'r', encoding='utf-8'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            _, stderr = process.communicate()
            print(f"[tts] Piper finished in {time.time()-t1:.2f}s | return code: {process.returncode}")
            if stderr:
                print(f"[tts] Piper stderr: {stderr.decode('utf-8', errors='replace')}")
            print(f"[tts] Output file: {output_file} | exists: {os.path.isfile(output_file)}")
        finally:
            os.remove(tmp_path)

        print(f"[tts] Total time: {time.time()-t0:.2f}s")
        return output_file

    def stream_tts(self, in_text, model="", model_dir=None, length=1, noise=1, width=1, sen_pause=1):
        t0 = time.time()
        if not model:
            model = self.model

        self.load_mod(instr=model, model_dir=model_dir)  # ✅ single call, with model_dir

        model_path = self.model_path  # ✅ from load_mod
        json_path = self.json_ob  # ✅ from load_mod

        print(f"[stream_tts] Writing temp input file...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False, encoding='utf-8') as tmp:
            tmp.write(in_text)
            tmp_path = tmp.name
        print(f"[stream_tts] Temp file: {tmp_path} | contents: '{in_text}'")

        try:
            print(f"[stream_tts] Launching piper subprocess...")
            t_launch = time.time()
            process = subprocess.Popen(
                [
                    self.piper_exe,
                    "--model", model_path,
                    "--config", json_path,
                    "--output-raw",
                    "--length_scale", str(length),
                    "--noise_scale", str(noise),
                    "--noise_w", str(width),
                    "--sentence_silence", str(sen_pause)
                ],
                stdin=open(tmp_path, 'r', encoding='utf-8'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"[stream_tts] Piper launched in {time.time() - t_launch:.3f}s | PID: {process.pid}")

            chunk_size = 4096
            chunk_count = 0
            total_bytes = 0
            t_first_chunk = None

            while True:
                data = process.stdout.read(chunk_size)
                if not data:
                    break
                if t_first_chunk is None:
                    t_first_chunk = time.time()
                    print(f"[stream_tts] First chunk received after {t_first_chunk - t0:.3f}s")
                chunk_count += 1
                total_bytes += len(data)
                self.buffer = data
                yield data

            print(f"[stream_tts] Stream done — {chunk_count} chunks | {total_bytes} bytes total")
            print(f"[stream_tts] Piper return code: {process.returncode}")

            stderr = process.stderr.read()
            if stderr:
                print(f"[stream_tts] Piper stderr: {stderr.decode('utf-8', errors='replace')}")

            if total_bytes == 0:
                print(f"[stream_tts] WARNING: 0 bytes received — piper produced no audio")

        finally:
            os.remove(tmp_path)
            print(f"[stream_tts] Total elapsed: {time.time() - t0:.2f}s")

    def get_sample_rate(self, model=""):
        if not model:
            model = self.model
        rate = self._get_sample_rate(model)
        print(f"[get_sample_rate] {model} -> {rate}Hz")
        return rate

    def save_set(self, model, length, noise, width, sen_pause):
        saved_dir = os.path.join(self.BASE_DIR, "saved")  # ✅ no os.getcwd()
        os.makedirs(saved_dir, exist_ok=True)
        set_json = {"model": model, "length": length, "noise": noise, "width": width, "pause": sen_pause}
        file_name = f'{model}__{length}__{noise}__{width}__{sen_pause}'.replace(".", "_")
        with open(os.path.join(saved_dir, f'{file_name}.json'), 'w') as f:
            f.write(json.dumps(set_json, indent=4))
        return f'{file_name}.json'

    def load_set(self, set_file):
        with open(set_file, 'r') as f:
            set_json = json.loads(f.read())
        return (
            set_json['model'],
            set_json['length'],
            set_json['noise'],
            set_json['width'],
            set_json['pause']
        )