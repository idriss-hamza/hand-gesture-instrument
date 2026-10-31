"""
song_library.py
Musical repertoire and note sequences for the Hand Gesture Instrument.
Includes Jingle Bells, Twinkle Twinkle Little Star, Ode to Joy, Happy Birthday, and Mary Had a Little Lamb.
Provides full text export to .txt files and clean terminal guide printing.
"""

from dataclasses import dataclass
import os
import sys
from typing import List, Dict, Optional


@dataclass
class SongNote:
    note: str              # e.g., 'E4', 'G4', 'C4', 'REST'
    duration: float        # duration in beats (1.0 = quarter note)
    lyric: str             # syllable / lyric
    gesture: str           # Target gesture name required to trigger note
    hand_hint: str = ""    # Visual hint for user (e.g., '2 Fingers (Peace)')


# Standard scale mapping from Note to Hand Gesture and Symbol
NOTE_TO_GESTURE_MAP = {
    'C4': ('Fist', 'Closed Fist (0 fingers / Do)'),
    'D4': ('Index Point', '1 Finger (Index / Re)'),
    'E4': ('Peace', '2 Fingers (Index+Middle / Mi)'),
    'F4': ('Three', '3 Fingers (Thumb+Ring in / Fa)'),
    'G4': ('Four', '4 Fingers (Thumb tucked / Sol)'),
    'A4': ('Open Palm', '5 Fingers (Open Palm / La)'),
    'B4': ('Thumbs Up', 'Thumbs Up (Ti)'),
    'C5': ('Pinch', 'Pinch / OK Sign (High Do)'),
    'D5': ('Rock On', 'Rock On (Index + Pinky / High Re)'),
    'E5': ('Hang Loose', 'Hang Loose / Shaka (Thumb + Pinky / High Mi)'),
    'F5': ('Gun', 'Gun / L-Shape (Thumb + Index / High Fa)'),
    'G5': ('Pinky Up', 'Pinky Only (High Sol)'),
    'A5': ('OK Sign', 'OK Sign (Circle / High La)'),
    'A#4': ('Rock On', 'Rock On (Index + Pinky / Bb4)'),
    'Bb4': ('Rock On', 'Rock On (Index + Pinky / Bb4)'),
    'F#4': ('Three', '3 Fingers (Fa#)'),
    'C#4': ('Fist', 'Closed Fist (Do# / Tilt Right)'),
    'D#4': ('Index Point', '1 Finger (Re# / Tilt Right)'),
    'G#4': ('Four', '4 Fingers (Sol# / Tilt Right)'),
    'REST': ('Rest', 'Open Hand Relaxed')
}

# Clean human-readable symbol names for screen display and guide
GESTURE_TO_SYMBOL_MAP = {
    'Fist': 'Closed Fist',
    'Index Point': '1 Finger (Index)',
    'Peace': '2 Fingers (Peace)',
    'Three': '3 Fingers',
    'Four': '4 Fingers',
    'Open Palm': 'Open Palm (5 Fingers)',
    'Thumbs Up': 'Thumbs Up',
    'Pinch': 'Pinch / Tip Touch',
    'Rock On': 'Rock On (Index + Pinky)',
    'Hang Loose': 'Hang Loose / Shaka',
    'Gun': 'Gun / L-Shape',
    'Pinky Up': 'Pinky Only',
    'OK Sign': 'OK Sign (Circle)',
    'Rest': 'Relaxed Hand',
    'None': 'None'
}

GESTURE_DESCRIPTIONS = {
    'Closed Fist': 'Curl all 5 fingers into a tight closed fist',
    '1 Finger (Index)': 'Extend only the index finger pointing up; curl all others',
    '2 Fingers (Peace)': 'Extend index and middle fingers in a "V" peace sign',
    '3 Fingers': 'Extend index, middle, and ring fingers (or thumb+index+mid)',
    '4 Fingers': 'Extend index, middle, ring, and pinky; tuck thumb across palm',
    'Open Palm (5 Fingers)': 'Spread all 5 fingers open wide facing the camera',
    'Thumbs Up': 'Curl all 4 fingers tightly and extend thumb straight up',
    'Pinch / Tip Touch': 'Touch the tips of your thumb and index finger together',
    'Rock On (Index + Pinky)': 'Extend index and pinky fingers; curl middle and ring',
    'Hang Loose / Shaka': 'Extend thumb and pinky outwards; curl middle three fingers',
    'Gun / L-Shape': 'Extend thumb and index finger at a 90 degree angle',
    'Pinky Only': 'Extend only the pinky finger; curl all other four fingers',
    'OK Sign (Circle)': 'Thumb and index tips touch into a circle; extend other 3 fingers'
}


@dataclass
class Song:
    title: str
    composer: str
    bpm: float
    recommended_instrument: str
    notes: List[SongNote]

    @property
    def total_notes(self) -> int:
        return sum(1 for n in self.notes if n.note != 'REST')


def _make_notes(raw_data: List[tuple]) -> List[SongNote]:
    """Helper to convert tuples (note, beats, lyric) into SongNote objects."""
    song_notes = []
    for item in raw_data:
        note, beats, lyric = item
        gest_info = NOTE_TO_GESTURE_MAP.get(note, ('Open Palm', 'Open Palm'))
        song_notes.append(
            SongNote(
                note=note,
                duration=float(beats),
                lyric=lyric,
                gesture=gest_info[0],
                hand_hint=gest_info[1]
            )
        )
    return song_notes


def get_jingle_bells() -> Song:
    """Jingle Bells Chorus with lyrics and gestures."""
    data = [
        # Part 1: "Jingle bells, jingle bells, jingle all the way"
        ('E4', 1.0, "Jin-"),
        ('E4', 1.0, "gle"),
        ('E4', 2.0, "bells,"),
        ('E4', 1.0, "jin-"),
        ('E4', 1.0, "gle"),
        ('E4', 2.0, "bells,"),
        ('E4', 1.0, "jin-"),
        ('G4', 1.0, "gle"),
        ('C4', 1.5, "all"),
        ('D4', 0.5, "the"),
        ('E4', 4.0, "way!"),

        # Part 2: "Oh what fun it is to ride in a one-horse open sleigh, hey!"
        ('F4', 1.0, "Oh"),
        ('F4', 1.0, "what"),
        ('F4', 1.5, "fun"),
        ('F4', 0.5, "it"),
        ('F4', 1.0, "is"),
        ('E4', 1.0, "to"),
        ('E4', 1.0, "ride"),
        ('E4', 0.5, "in"),
        ('E4', 0.5, "a"),
        ('E4', 1.0, "one-"),
        ('D4', 1.0, "horse"),
        ('D4', 1.0, "o-"),
        ('E4', 1.0, "pen"),
        ('D4', 2.0, "sleigh,"),
        ('G4', 2.0, "hey!"),

        # Part 3: "Jingle bells, jingle bells, jingle all the way"
        ('E4', 1.0, "Jin-"),
        ('E4', 1.0, "gle"),
        ('E4', 2.0, "bells,"),
        ('E4', 1.0, "jin-"),
        ('E4', 1.0, "gle"),
        ('E4', 2.0, "bells,"),
        ('E4', 1.0, "jin-"),
        ('G4', 1.0, "gle"),
        ('C4', 1.5, "all"),
        ('D4', 0.5, "the"),
        ('E4', 4.0, "way!"),

        # Part 4: "Oh what fun it is to ride in a one-horse open sleigh!"
        ('F4', 1.0, "Oh"),
        ('F4', 1.0, "what"),
        ('F4', 1.5, "fun"),
        ('F4', 0.5, "it"),
        ('F4', 1.0, "is"),
        ('E4', 1.0, "to"),
        ('E4', 1.0, "ride"),
        ('E4', 0.5, "in"),
        ('E4', 0.5, "a"),
        ('G4', 1.0, "one-"),
        ('G4', 1.0, "horse"),
        ('F4', 1.0, "o-"),
        ('D4', 1.0, "pen"),
        ('C4', 4.0, "sleigh!"),
    ]
    return Song(
        title="Jingle Bells",
        composer="James Lord Pierpont",
        bpm=120.0,
        recommended_instrument="bell",
        notes=_make_notes(data)
    )


def get_twinkle_twinkle() -> Song:
    """Twinkle Twinkle Little Star melody."""
    data = [
        ('C4', 1.0, "Twin-"), ('C4', 1.0, "kle,"), ('G4', 1.0, "twin-"), ('G4', 1.0, "kle,"),
        ('A4', 1.0, "lit-"), ('A4', 1.0, "tle"), ('G4', 2.0, "star,"),
        ('F4', 1.0, "How"), ('F4', 1.0, "I"), ('E4', 1.0, "won-"), ('E4', 1.0, "der"),
        ('D4', 1.0, "what"), ('D4', 1.0, "you"), ('C4', 2.0, "are!"),
        ('G4', 1.0, "Up"), ('G4', 1.0, "a-"), ('F4', 1.0, "bove"), ('F4', 1.0, "the"),
        ('E4', 1.0, "world"), ('E4', 1.0, "so"), ('D4', 2.0, "high,"),
        ('G4', 1.0, "Like"), ('G4', 1.0, "a"), ('F4', 1.0, "dia-"), ('F4', 1.0, "mond"),
        ('E4', 1.0, "in"), ('E4', 1.0, "the"), ('D4', 2.0, "sky,"),
        ('C4', 1.0, "Twin-"), ('C4', 1.0, "kle,"), ('G4', 1.0, "twin-"), ('G4', 1.0, "kle,"),
        ('A4', 1.0, "lit-"), ('A4', 1.0, "tle"), ('G4', 2.0, "star,"),
        ('F4', 1.0, "How"), ('F4', 1.0, "I"), ('E4', 1.0, "won-"), ('E4', 1.0, "der"),
        ('D4', 1.0, "what"), ('D4', 1.0, "you"), ('C4', 2.0, "are!")
    ]
    return Song(
        title="Twinkle Twinkle Little Star",
        composer="Traditional",
        bpm=100.0,
        recommended_instrument="piano",
        notes=_make_notes(data)
    )


def get_ode_to_joy() -> Song:
    """Beethoven's Ode to Joy theme."""
    data = [
        ('E4', 1.0, "Joy"), ('E4', 1.0, "bright"), ('F4', 1.0, "spark"), ('G4', 1.0, "of"),
        ('G4', 1.0, "di-"), ('F4', 1.0, "vi-"), ('E4', 1.0, "ni-"), ('D4', 1.0, "ty,"),
        ('C4', 1.0, "Daugh-"), ('C4', 1.0, "ter"), ('D4', 1.0, "of"), ('E4', 1.0, "E-"),
        ('E4', 1.5, "ly-"), ('D4', 0.5, "si-"), ('D4', 2.0, "um,"),
        ('E4', 1.0, "We"), ('E4', 1.0, "trans-"), ('F4', 1.0, "port-"), ('G4', 1.0, "ed"),
        ('G4', 1.0, "by"), ('F4', 1.0, "thy"), ('E4', 1.0, "beau-"), ('D4', 1.0, "ty,"),
        ('C4', 1.0, "To"), ('C4', 1.0, "thy"), ('D4', 1.0, "ho-"), ('E4', 1.0, "ly"),
        ('D4', 1.5, "tem-"), ('C4', 0.5, "ple"), ('C4', 2.0, "come!")
    ]
    return Song(
        title="Ode to Joy",
        composer="Ludwig van Beethoven",
        bpm=110.0,
        recommended_instrument="synth",
        notes=_make_notes(data)
    )


def get_happy_birthday() -> Song:
    """Happy Birthday to You."""
    data = [
        ('C4', 0.75, "Hap-"), ('C4', 0.25, "py"), ('D4', 1.0, "birth-"), ('C4', 1.0, "day"), ('F4', 1.0, "to"), ('E4', 2.0, "you,"),
        ('C4', 0.75, "Hap-"), ('C4', 0.25, "py"), ('D4', 1.0, "birth-"), ('C4', 1.0, "day"), ('G4', 1.0, "to"), ('F4', 2.0, "you,"),
        ('C4', 0.75, "Hap-"), ('C4', 0.25, "py"), ('C5', 1.0, "birth-"), ('A4', 1.0, "day"), ('F4', 1.0, "dear"), ('E4', 1.0, "friend,"), ('D4', 2.0, ""),
        ('Bb4', 0.75, "Hap-"), ('Bb4', 0.25, "py"), ('A4', 1.0, "birth-"), ('F4', 1.0, "day"), ('G4', 1.0, "to"), ('F4', 2.0, "you!")
    ]
    return Song(
        title="Happy Birthday",
        composer="Patty and Mildred J. Hill",
        bpm=100.0,
        recommended_instrument="violin",
        notes=_make_notes(data)
    )


def get_mary_had_a_little_lamb() -> Song:
    """Mary Had a Little Lamb."""
    data = [
        ('E4', 1.0, "Ma-"), ('D4', 1.0, "ry"), ('C4', 1.0, "had"), ('D4', 1.0, "a"),
        ('E4', 1.0, "lit-"), ('E4', 1.0, "tle"), ('E4', 2.0, "lamb,"),
        ('D4', 1.0, "lit-"), ('D4', 1.0, "tle"), ('D4', 2.0, "lamb,"),
        ('E4', 1.0, "lit-"), ('G4', 1.0, "tle"), ('G4', 2.0, "lamb!"),
        ('E4', 1.0, "Ma-"), ('D4', 1.0, "ry"), ('C4', 1.0, "had"), ('D4', 1.0, "a"),
        ('E4', 1.0, "lit-"), ('E4', 1.0, "tle"), ('E4', 1.0, "lamb,"), ('E4', 1.0, "its"),
        ('D4', 1.0, "fleece"), ('D4', 1.0, "was"), ('E4', 1.0, "white"), ('D4', 1.0, "as"),
        ('C4', 4.0, "snow!")
    ]
    return Song(
        title="Mary Had a Little Lamb",
        composer="Traditional",
        bpm=110.0,
        recommended_instrument="bell",
        notes=_make_notes(data)
    )


def get_fur_elise() -> Song:
    """Beethoven's Für Elise (Theme) showcasing extended octave notes (E5, D5, C5, B4, A4)."""
    data = [
        ('E5', 0.5, "Für"), ('D#4', 0.5, "E-"), ('E5', 0.5, "lise,"), ('D#4', 0.5, "Für"),
        ('E5', 0.5, "E-"), ('B4', 0.5, "lise,"), ('D5', 0.5, "so"), ('C5', 0.5, "sweet,"),
        ('A4', 2.0, "fair!"),
        ('C4', 0.5, "In"), ('E4', 0.5, "the"), ('A4', 0.5, "morn-"), ('B4', 2.0, "ing,"),
        ('E4', 0.5, "mu-"), ('G#4', 0.5, "sic"), ('B4', 0.5, "fills"), ('C5', 2.0, "air!"),
        ('E5', 0.5, "Für"), ('D#4', 0.5, "E-"), ('E5', 0.5, "lise,"), ('D#4', 0.5, "Für"),
        ('E5', 0.5, "E-"), ('B4', 0.5, "lise,"), ('D5', 0.5, "so"), ('C5', 0.5, "sweet,"),
        ('A4', 2.5, "fair!")
    ]
    return Song(
        title="Für Elise",
        composer="Ludwig van Beethoven",
        bpm=110.0,
        recommended_instrument="piano",
        notes=_make_notes(data)
    )


SONG_CATALOG: Dict[str, Song] = {
    'jingle_bells': get_jingle_bells(),
    'twinkle_twinkle': get_twinkle_twinkle(),
    'ode_to_joy': get_ode_to_joy(),
    'happy_birthday': get_happy_birthday(),
    'mary_had_a_little_lamb': get_mary_had_a_little_lamb(),
    'fur_elise': get_fur_elise(),
}


def get_song(key: str) -> Optional[Song]:
    return SONG_CATALOG.get(key.lower().replace(' ', '_'))


def get_available_songs() -> List[str]:
    return list(SONG_CATALOG.keys())


def generate_text_guide() -> str:
    """Generate the complete, user-facing plain text reference guide."""
    lines = []
    lines.append("=" * 80)
    lines.append("       HAND GESTURE MUSICAL INSTRUMENT - GESTURES & SONGS REFERENCE GUIDE")
    lines.append("=" * 80)
    lines.append("")
    lines.append("1. GESTURE & SYMBOL REFERENCE TABLE")
    lines.append("-" * 80)
    lines.append(f"{'NOTE':<6} {'GESTURE':<14} {'SYMBOL ON SCREEN':<25} {'HOW TO DO THE GESTURE'}")
    lines.append("-" * 80)

    rows = [
        ('C4', 'Fist', 'Closed Fist', 'Closed Fist'),
        ('D4', 'Index Point', '1 Finger (Index)', '1 Finger (Index)'),
        ('E4', 'Peace', '2 Fingers (Peace)', '2 Fingers (Peace)'),
        ('F4', 'Three', '3 Fingers', '3 Fingers'),
        ('G4', 'Four', '4 Fingers', '4 Fingers'),
        ('A4', 'Open Palm', 'Open Palm (5 Fingers)', 'Open Palm (5 Fingers)'),
        ('B4', 'Thumbs Up', 'Thumbs Up', 'Thumbs Up'),
        ('C5', 'Pinch', 'Pinch / Tip Touch', 'Pinch / Tip Touch'),
        ('D5', 'Rock On', 'Rock On (Index + Pinky)', 'Rock On (Index + Pinky)'),
        ('E5', 'Hang Loose', 'Hang Loose / Shaka', 'Hang Loose / Shaka'),
        ('F5', 'Gun', 'Gun / L-Shape', 'Gun / L-Shape'),
        ('G5', 'Pinky Up', 'Pinky Only', 'Pinky Only'),
        ('A5', 'OK Sign', 'OK Sign (Circle)', 'OK Sign (Circle)'),
    ]

    for note, gest, sym, desc_key in rows:
        desc = GESTURE_DESCRIPTIONS.get(desc_key, "")
        lines.append(f"{note:<6} {gest:<14} {sym:<25} {desc}")

    lines.append("-" * 80)
    lines.append("")
    lines.append("2. HAND ROTATION SOUND CONTROLS (FIST CLUTCH SYSTEM)")
    lines.append("-" * 80)
    lines.append("Rotate / tilt your hand clockwise or counter-clockwise in front of the camera:")
    lines.append("NOTE: Rotary controls only activate when you form a CLOSED FIST!")
    lines.append("When your hand is OPEN (or doing any other gesture), rotation has NO effect,")
    lines.append("allowing you to comfortably return your hand to center (50% marker) while")
    lines.append("keeping your dialed volume and pitch shift locked in place!")
    lines.append("")
    lines.append("  [RIGHT HAND] -> PITCH SHIFT / BEND:")
    lines.append("    * Close Fist and rotate to adjust pitch offset (-12 to +12 semitones).")
    lines.append("    * Upright (0 deg): Natural base note (0 semitones).")
    lines.append("    * Tilt Right (Clockwise): Shifts pitch UP (+1 to +12 semitones / sharps).")
    lines.append("    * Tilt Left (Counter-Clockwise): Shifts pitch DOWN (-1 to -12 semitones / flats).")
    lines.append("    * Open your hand to LOCK the pitch shift and play notes freely!")
    lines.append("")
    lines.append("  [LEFT HAND] -> MASTER VOLUME DIAL:")
    lines.append("    * Close Fist and rotate to adjust master volume (0% to 100%).")
    lines.append("    * Tilt Left (-60 deg): 0% Volume (Mute).")
    lines.append("    * Upright (0 deg): 65% Volume (Standard listening level).")
    lines.append("    * Tilt Right (+60 deg): 100% Volume (Maximum loudness).")
    lines.append("    * Open your hand to LOCK the volume (move hand back to 50% marker freely)!")
    lines.append("")
    lines.append("3. MINIMALIST ON-SCREEN DISPLAY & MODES")
    lines.append("-" * 80)
    lines.append("  * Video & Hand Skeleton: Clean, unobstructed camera view with glowing landmarks.")
    lines.append("  * Song Guide Mode [1]: Shows target note & symbol in vivid ORANGE. Guided play-along tutorial.")
    lines.append("  * Free Instrument Mode [2/F]: Freely plays whatever gesture is shown on screen without guidance.")
    lines.append("    Displays the detected note in vivid ORANGE with its symbol, pitch shift and volume controls.")
    lines.append("  * Rotation Meters: Real-time Pitch Shift (Right Hand) and Volume % (Left Hand).")
    lines.append("  * Digital Camera Top Cropping: Press [ or ] to dynamically fine-tune framing (-5% / +5%) to eliminate obstacles.")
    lines.append("  * No distracting song highway or lyric clutter on the camera stream!")
    lines.append("")
    lines.append("4. SONGS CATALOG & STEP-BY-STEP NOTE GUIDES")
    lines.append("-" * 80)

    for song_id, song in SONG_CATALOG.items():
        lines.append(f"\n==================== [ {song.title.upper()} ] ====================")
        lines.append(f"Composer: {song.composer}  |  BPM: {int(song.bpm)}  |  Instrument: {song.recommended_instrument.upper()}")
        lines.append(f"Total Notes: {song.total_notes}")
        lines.append("-" * 80)
        lines.append(f"{'#':<4} {'NOTE':<6} {'SYMBOL / GESTURE':<28} {'BEATS':<7} {'LYRIC'}")
        lines.append("-" * 80)

        for idx, sn in enumerate(song.notes):
            sym = GESTURE_TO_SYMBOL_MAP.get(sn.gesture, sn.gesture)
            lines.append(f"{idx + 1:<4} {sn.note:<6} {sym:<28} {sn.duration:<7.1f} {sn.lyric}")

    lines.append("\n" + "=" * 80 + "\n")
    return "\n".join(lines)


def save_guide_to_file(path: str = "SONG_LIBRARY_AND_GESTURES.txt"):
    """Write the guide to a disk file."""
    content = generate_text_guide()
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[SongLibrary] Guide successfully written to {path}")


def print_library_to_terminal(song_key: Optional[str] = None):
    """Print the complete guide or specific song to the terminal."""
    if song_key:
        song = get_song(song_key)
        if not song:
            print(f"[SongLibrary] Unknown song: '{song_key}'. Available: {get_available_songs()}")
            return
        print(f"\n=== {song.title.upper()} ({song.recommended_instrument.upper()}) ===")
        print(f"{'#':<4} {'NOTE':<6} {'SYMBOL / GESTURE':<28} {'LYRIC'}")
        print("-" * 50)
        for idx, sn in enumerate(song.notes):
            sym = GESTURE_TO_SYMBOL_MAP.get(sn.gesture, sn.gesture)
            print(f"{idx + 1:<4} {sn.note:<6} {sym:<28} {sn.lyric}")
        print("-" * 50 + "\n")
    else:
        print(generate_text_guide())


if __name__ == '__main__':
    save_guide_to_file()
    print_library_to_terminal()
