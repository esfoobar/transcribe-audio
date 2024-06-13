import os
import sys
import time
from pydub import AudioSegment
import speech_recognition as sr


def convert_to_wav(audio_file_path):
    if audio_file_path.endswith(".wav"):
        return audio_file_path
    print("Converting audio file to WAV format...")
    sound = AudioSegment.from_file(audio_file_path)
    wav_file_path = audio_file_path.rsplit(".", 1)[0] + ".wav"
    sound.export(wav_file_path, format="wav")
    print("Conversion complete.")
    return wav_file_path


def split_audio(audio_file_path, chunk_length_ms=60000):
    print("Splitting audio into chunks...")
    audio = AudioSegment.from_wav(audio_file_path)
    base_name = os.path.basename(audio_file_path).rsplit(".", 1)[0]
    chunk_folder = os.path.join(os.path.dirname(audio_file_path), f"{base_name}_chunks")
    os.makedirs(chunk_folder, exist_ok=True)

    chunks = [
        audio[i : i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)
    ]
    chunk_files = []
    for i, chunk in enumerate(chunks):
        chunk_file_path = os.path.join(chunk_folder, f"{base_name}_chunk{i}.wav")
        chunk.export(chunk_file_path, format="wav")
        chunk_files.append(chunk_file_path)
        print(f"Created chunk {i+1}/{len(chunks)}: {chunk_file_path}")
    print("Audio splitting complete.")
    return chunk_files, chunk_folder


def transcribe_audio_chunk(audio_file_path, chunk_index, total_chunks):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)

    retries = 3
    for attempt in range(retries):
        try:
            transcript = recognizer.recognize_google(audio_data)
            print(f"Transcription of chunk {chunk_index+1}/{total_chunks} complete.")
            return transcript
        except (sr.RequestError, sr.UnknownValueError) as e:
            print(
                f"Attempt {attempt + 1} for chunk {chunk_index+1}/{total_chunks} failed: {e}"
            )
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise e


def transcribe_audio(audio_file_path):
    audio_file_path = convert_to_wav(audio_file_path)
    chunk_files, chunk_folder = split_audio(audio_file_path)

    full_transcript = ""
    total_chunks = len(chunk_files)
    for i, chunk_file in enumerate(chunk_files):
        transcript = transcribe_audio_chunk(chunk_file, i, total_chunks)
        full_transcript += transcript + " "

    # Clean up chunk files and folder after processing
    for chunk_file in chunk_files:
        os.remove(chunk_file)
    os.rmdir(chunk_folder)

    return full_transcript.strip()


def save_transcript_to_desktop(transcript, filename="transcription.txt"):
    desktop_path = os.path.join(os.path.join(os.path.expanduser("~")), "Desktop")
    file_path = os.path.join(desktop_path, filename)

    with open(file_path, "w") as file:
        file.write(transcript)

    print(f"Transcription saved to {file_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transcribe_audio.py <audio_file_path>")
        sys.exit(1)

    audio_file_path = sys.argv[1]

    if not os.path.isfile(audio_file_path):
        print(f"The file {audio_file_path} does not exist.")
        sys.exit(1)

    try:
        print(f"Starting transcription for {audio_file_path}...")
        transcript = transcribe_audio(audio_file_path)
        save_transcript_to_desktop(transcript)
        print("Transcription complete.")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
