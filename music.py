import numpy as np
from midiutil import MIDIFile
from scipy.signal import convolve

TEMPO_RANGE = (60, 180)
SEMITONES = 12
MIDI_INSTRUMENTS = 128

EMBED_SLICES = {
    'tempo': slice(0, 1),
    'key': slice(1, 2),
    'melody_pattern': slice(2, 18),
    'melody_instrument': slice(18, 19),
    'bass_pattern': slice(19, 35),
    'bass_instrument': slice(35, 36),
    'harmony_pattern': slice(36, 52),
    'harmony_instrument': slice(52, 53),
    'percussion_pattern': slice(53, 69),
    'percussion_instrument': slice(69, 70)
}

def pad_embedding(embed, target_size=70):
    """
    Pad or trim the embedding to the target size with zeros if needed.

    Args:
        embed (np.ndarray): The embedding array.
        target_size (int): The desired length of the embedding.
    Returns:
        np.ndarray: The padded or trimmed embedding.
    """
    if len(embed) >= target_size:
        return embed[:target_size]
    return np.pad(embed, (0, target_size - len(embed)), mode='constant')

def normalize_embedding(embed):
    """
    Normalize the embedding to the [0, 1] range.

    Args:
        embed (np.ndarray): The embedding array.
    Returns:
        np.ndarray: The normalized embedding.
    """
    embed_min = np.min(embed)
    embed_max = np.max(embed)
    if embed_max == embed_min:
        return np.zeros_like(embed)
    return (embed - embed_min) / (embed_max - embed_min)

def extract_rhythm_patterns(embed_norm):
    """
    Extract rhythm patterns for each instrument from the normalized embedding.

    Args:
        embed_norm (np.ndarray): The normalized embedding.
    Returns:
        dict: Dictionary with rhythm patterns for melody, bass, harmony, and percussion.
    """
    return {k.split('_')[0]: embed_norm[v] for k, v in EMBED_SLICES.items() if 'pattern' in k}

def extract_song_structure(_):
    """
    Return a fixed song structure.

    Returns:
        dict: Dictionary with song structure sections and their lengths.
    """
    return {'intro_length': 4, 'verse_length': 8, 'chorus_length': 8, 'outro_length': 4}

def calculate_complexity_from_text_length(text_length):
    """
    Calculate a complexity level based on the length of the input text.

    Args:
        text_length (int): The length of the input text.
    Returns:
        int: Complexity level from 1 (simple) to 5 (complex).
    """
    if text_length is None:
        return 4
    if text_length < 100:
        return 1
    if text_length < 300:
        return 2
    if text_length < 600:
        return 3
    if text_length < 1000:
        return 4
    return 5

def get_active_instruments(complexity_level):
    """
    Return a list of active instruments based on the complexity level.

    Args:
        complexity_level (int): The complexity level.
    Returns:
        list: List of instrument names to be used.
    """
    levels = [
        ['melody'],
        ['melody', 'bass'],
        ['melody', 'bass', 'harmony'],
        ['melody', 'bass', 'harmony', 'percussion']
    ]
    return levels[min(complexity_level - 1, 3)]

def scale_note(base_note, offset, scale):
    """
    Scale a note by a given offset within a musical scale.

    Args:
        base_note (int): The base MIDI note.
        offset (int): The offset to apply.
        scale (list): The scale intervals.
    Returns:
        int: The resulting MIDI note.
    """
    octave, index = divmod(offset, len(scale))
    return base_note + octave * 12 + scale[index]

def generate_melody(rhythm, pattern, key, threshold=0.4, desired_length=16, embed=None):
    """
    Generate a melody sequence based on rhythm, pattern, key, and embedding features.

    Args:
        rhythm (np.ndarray): Rhythm pattern.
        pattern (np.ndarray): Pattern for the melody.
        key (int): Key signature.
        threshold (float): Threshold for note activation.
        desired_length (int): Desired number of beats.
        embed (np.ndarray, optional): Embedding for additional features.
    Returns:
        list: List of (note, start_beat, duration) tuples.
    """
    MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
    base_note = 60 + key
    notes = []
    i = 0
    current_beat = 0

    if embed is not None:
        rhythm_shape = int((embed[10] * 3) % 3)
        rhythm_probs = [
            [0.2, 0.6, 0.2],
            [0.5, 0.3, 0.2],
            [0.1, 0.2, 0.7]
        ][rhythm_shape]
        transposition = int((embed[20] * 4) % 4) - 2
        inversion = bool(int((embed[25] * 10) % 2))
    else:
        rhythm_probs = [0.2, 0.6, 0.2]
        transposition = 0
        inversion = False

    motif_base = pattern[:8]
    variations = [motif_base + 0.02 * np.random.randn(len(motif_base)) for _ in range(4)]
    if inversion:
        for j in range(1, 4, 2):
            variations[j] = variations[j][::-1]
    variations = [convolve(v, [0.25, 0.5, 0.25], mode='same') for v in variations]
    final_pattern = np.concatenate(variations)
    curve = 0.5 + 0.5 * np.sin(np.linspace(0, 3 * np.pi, len(final_pattern)))
    pattern_shaped = (final_pattern + curve) / 2

    while current_beat < desired_length:
        value = pattern_shaped[i % len(pattern_shaped)]
        offset = int((max(value, threshold) - threshold) * len(MAJOR_SCALE) * 2)
        note = scale_note(base_note + transposition, offset, MAJOR_SCALE)
        note = np.clip(note, 24, 127)
        duration = np.random.choice([0.25, 0.5, 0.75], p=rhythm_probs)
        notes.append((note, current_beat, duration))
        current_beat += duration
        i += 1
    return notes

def extract_instruments(embed_norm, complexity_level=4):
    """
    Extract instrument patterns and MIDI program numbers from the embedding.

    Args:
        embed_norm (np.ndarray): The normalized embedding.
        complexity_level (int): The complexity level.
    Returns:
        dict: Dictionary of instrument configurations.
    """
    def safe_instrument(slice_):
        return int(embed_norm[slice_] * (MIDI_INSTRUMENTS - 1))
    all_instruments = {
        'melody': {
            'pattern': embed_norm[EMBED_SLICES['melody_pattern']],
            'instrument': safe_instrument(EMBED_SLICES['melody_instrument']),
            'channel': 0
        },
        'bass': {
            'pattern': embed_norm[EMBED_SLICES['bass_pattern']],
            'instrument': safe_instrument(EMBED_SLICES['bass_instrument']),
            'channel': 1
        },
        'harmony': {
            'pattern': embed_norm[EMBED_SLICES['harmony_pattern']],
            'instrument': safe_instrument(EMBED_SLICES['harmony_instrument']),
            'channel': 2
        },
        'percussion': {
            'pattern': embed_norm[EMBED_SLICES['percussion_pattern']],
            'instrument': 0,  # Standard percussion
            'channel': 9
        }
    }
    active = get_active_instruments(complexity_level)
    return {k: v for k, v in all_instruments.items() if k in active}

def generate_music_embed(embed, text_length=None):
    """
    Generate all music features from an embedding and optional text length.

    Args:
        embed (np.ndarray): The embedding array.
        text_length (int, optional): The length of the input text.
    Returns:
        dict: Dictionary of all music features.
    """
    embed_padded = pad_embedding(embed)
    embed_norm = normalize_embedding(embed_padded)
    complexity_level = calculate_complexity_from_text_length(text_length)
    tempo = int(TEMPO_RANGE[0] + embed_norm[EMBED_SLICES['tempo']] * (TEMPO_RANGE[1] - TEMPO_RANGE[0]))
    key_signature = int(embed_norm[EMBED_SLICES['key']] * SEMITONES)
    return {
        'tempo': tempo,
        'key': key_signature,
        'instruments': extract_instruments(embed_norm, complexity_level),
        'rhythm_patterns': extract_rhythm_patterns(embed_norm),
        'song_structure': extract_song_structure(embed_norm),
        'complexity_level': complexity_level,
        'embed': embed_norm
    }

def render_music(music_features, output_path, text_length=None):
    """
    Render music features to a MIDI file and save it to the given path.

    Args:
        music_features (dict or np.ndarray): Music features or embedding.
        output_path (str): Path to save the MIDI file.
        text_length (int, optional): The length of the input text.
    Returns:
        None
    """
    if isinstance(music_features, np.ndarray):
        music_features = generate_music_embed(music_features, text_length)
    tempo = music_features['tempo']
    key = music_features['key']
    instruments = music_features['instruments']
    rhythm_patterns = music_features['rhythm_patterns']
    embed = music_features['embed']
    complexity_level = music_features.get('complexity_level', 4)
    midi_file = MIDIFile(len(instruments))
    midi_file.addTempo(0, 0, tempo)
    for instrument_name, instrument_data in instruments.items():
        channel = instrument_data['channel']
        pattern = instrument_data['pattern']
        instrument_id = instrument_data['instrument']
        if channel != 9:
            midi_file.addProgramChange(0, channel, 0, instrument_id)
        rhythm = rhythm_patterns.get(instrument_name, pattern)
        melody = generate_melody(rhythm, pattern, key, desired_length=16, embed=embed)
        velocity = max(30, 100 - complexity_level * 5)
        for note, beat_time, duration in melody:
            if instrument_name == 'bass':
                note -= 12
            elif instrument_name == 'harmony':
                note += 7
            midi_file.addNote(0, channel, int(note), beat_time, duration, velocity)
    with open(output_path, 'wb') as f:
        midi_file.writeFile(f)
    print(f"✅ Music saved to: {output_path}")
    print(f"🎵 Complexity level: {complexity_level} ({len(instruments)} instruments)")



