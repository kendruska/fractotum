# Fractotum

**Fractotum** is an experimental system for generating rich, unique, and highly complex procedural art from literary text — without using generative AI, pretrained visual models, or external datasets. It is a reflection on algorithmic creativity through mathematical processes, inspired by fractals, text embeddings, and the internal geometry of language.

This project is 100% code-driven: it does not remix existing artworks or learn from other artists. Everything is generated procedurally from scratch, based on the mathematical transformation of the input text.

---

## ✨ What It Does

You provide a `.txt` file with a literary fragment (10+ lines recommended). Fractotum:

1. Extracts a semantic **embedding** of the text using a transformer model.
2. Maps the embedding to **visual parameters**: complex constants, zoom, iteration count, density, and color palettes.
3. Generates a unique **fractal visual** (still or animated) using:
   - Julia set dynamics
   - Distortion fields
   - Gravitational centers
   - Rotational flows
   - Ray-based lighting
   - Particle-style accumulation
4. (New!) Generates a unique **procedural audio (MIDI)** file from the same text embedding, mapping semantic features to musical structure, instruments, and melody.

---

## 📁 Project Structure

```
Fractotum/
├── main.py                   # Entry point
├── text_processing.py        # Text-to-embedding + param mapping
├── procedural_generation.py  # Image/video generation with procedural fractals
├── music.py                  # Audio (MIDI) generation from text embedding
├── requirements.txt          # All required Python dependencies
```

---

## ⚙️ Installation

You'll need Python 3.8+.

All required dependencies are listed in `requirements.txt`. To install them, run:

```bash
pip install -r requirements.txt
```

Or, if you prefer, you can install the main libraries individually:

```bash
pip install sentence-transformers matplotlib numpy scipy midiutil pillow numba
```

You also need `ffmpeg` if you want to export animated `.mp4` files.

### 🪟 On Windows

#### ✅ Option 1: Using Chocolatey (recommended)

If you have [Chocolatey](https://chocolatey.org/install) installed, just run:

```bash
choco install ffmpeg
```

Then restart your terminal and check:

```bash
ffmpeg -version
```

#### 📦 Option 2: Manual installation

1. Download from: <https://www.gyan.dev/ffmpeg/builds/>
2. Extract the ZIP (e.g., `ffmpeg-release-essentials.zip`) to a folder like `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your **System PATH**:
   - Go to: Control Panel → System → Advanced → Environment Variables
   - Edit `Path` and add a new entry: `C:\ffmpeg\bin`
4. Restart your terminal or IDE

---

## 🚀 Usage

### 🖼 Generate an image

```bash
python main.py --input my_text.txt --mode image
```

Output: `my_text_fractotum.png`

### 🎞 Generate a video (animated)

```bash
python main.py --input my_text.txt --mode video
```

Output: `my_text_fractotum.mp4`

### 🎵 Generate audio (MIDI)

```bash
python main.py --input my_text.txt --mode audio
```

Output: `my_text_fractotum.mid`

---

## 🧪 Test Example

### 📄 Input Text

```
En un lugar de la Mancha, de cuyo nombre no quiero
acordarme, no ha mucho tiempo que vivía un hidalgo de los de
lanza en astillero, adarga antigua, rocín flaco y galgo
corredor. Una olla de algo más vaca que carnero, salpicón
las más noches, duelos y quebrantos los sábados, lantejas
los viernes, algún palomino de añadidura los domingos,
consumían las tres partes de su hacienda. El resto della
concluían sayo de velarte, calzas de velludo para las
fiestas, con sus pantuflos de lo mesmo, y los días de
entresemana se honraba con su vellorí de lo más fino. Tenía
en su casa una ama que pasaba de los cuarenta, y una sobrina
que no llegaba a los veinte, y un mozo de campo y plaza, que
así ensillaba el rocín como tomaba la podadera. Frisaba la
edad de nuestro hidalgo con los cincuenta años; era de
complexión recia, seco de carnes, enjuto de rostro, gran
madrugador y amigo de la caza. Quieren decir que tenía el
sobrenombre de Quijada, o Quesada, que en esto hay alguna
diferencia en los autores que deste caso escriben; aunque
por conjeturas verosímiles se deja entender que se llamaba
Quijana. Pero esto importa poco a nuestro cuento: basta que
en la narración dél no se salga un punto de la verdad.
```

### 🖼️ Generated Image

![Example Output](test_data/test_fractotum.png)

### 🎵 Generated Audio (MIDI)

The command above will also generate a MIDI file, which you can open in any DAW or MIDI player.

---

## 💡 Philosophy

This project is a creative experiment in **algorithmic art without AI synthesis**.

I deliberately avoid:

- Generative adversarial networks (GANs)
- Diffusion models
- Any model trained to imitate human art

Instead, Fractotum explores how **pure mathematics, text embeddings, and iteration** can create emotionally resonant visuals and music that emerge from **semantic structure**, not learned imitation.

It doesn’t borrow from other artists, datasets, or visuals. It aims to explore how far we can push mathematical beauty using:

- Semantic meaning  
- Procedural geometry  
- Pure computation  

The result is something **unique to each text**, with a visual and musical identity drawn only from your words and mathematical logic.

---

## 🔧 Future Ideas

- Fractal families across a full poem or book
- Multi-mode styles (chaotic, soft, organic...)
- Real-time interactive visualizer
- Export to high-res prints or vector art
- CLI support for multiple `.txt` batch rendering
- More advanced audio synthesis

---

## 📄 License

This is a personal and artistic experiment. You're welcome to explore, remix, or extend it, as long as you respect the original intent: **procedural creation, not generative reuse**.

<<<<<<< HEAD

=======
## 🙏 Acknowledgments

🤗 Huge thanks to **Aday**, who volunteered as tribute to test this procedural experiment. 
>>>>>>> 7b519eb8c9c84fb80b56103a0927e9a54cc786ee
