"""
Unit tests for songs repertoire and notes sequencing.
"""

import pytest
from song_library import (
    SONG_CATALOG,
    get_song,
    get_jingle_bells,
    get_twinkle_twinkle,
    get_ode_to_joy,
    get_happy_birthday,
    NOTE_TO_GESTURE_MAP
)


def test_jingle_bells_structure():
    jb = get_jingle_bells()
    assert jb.title == "Jingle Bells"
    assert jb.bpm == 120.0
    assert jb.recommended_instrument == "bell"
    assert len(jb.notes) > 20

    # Verify first 3 notes are E4 (Jin-gle bells)
    assert jb.notes[0].note == "E4"
    assert jb.notes[1].note == "E4"
    assert jb.notes[2].note == "E4"

    # Verify next 3 are also E4 (jin-gle bells)
    assert jb.notes[3].note == "E4"
    assert jb.notes[4].note == "E4"
    assert jb.notes[5].note == "E4"

    # Next are E4, G4, C4, D4, E4 (jingle all the way)
    assert jb.notes[6].note == "E4"
    assert jb.notes[7].note == "G4"
    assert jb.notes[8].note == "C4"
    assert jb.notes[9].note == "D4"
    assert jb.notes[10].note == "E4"


def test_all_catalog_songs():
    expected_songs = ['jingle_bells', 'twinkle_twinkle', 'ode_to_joy', 'happy_birthday']
    for key in expected_songs:
        song = get_song(key)
        assert song is not None, f"Missing song: {key}"
        assert song.bpm > 40.0
        assert len(song.notes) > 0

        # Check each note is valid
        for n in song.notes:
            assert n.duration > 0
            assert len(n.gesture) > 0
            assert len(n.lyric) >= 0


def test_note_to_gesture_map():
    assert NOTE_TO_GESTURE_MAP['C4'][0] == 'Fist'
    assert NOTE_TO_GESTURE_MAP['D4'][0] == 'Index Point'
    assert NOTE_TO_GESTURE_MAP['E4'][0] == 'Peace'
    assert NOTE_TO_GESTURE_MAP['F4'][0] == 'Three'
    assert NOTE_TO_GESTURE_MAP['G4'][0] == 'Four'
    assert NOTE_TO_GESTURE_MAP['A4'][0] == 'Open Palm'
    assert NOTE_TO_GESTURE_MAP['B4'][0] == 'Thumbs Up'
    assert NOTE_TO_GESTURE_MAP['C5'][0] == 'Pinch'
    assert NOTE_TO_GESTURE_MAP['Bb4'][0] == 'Rock On'
    assert NOTE_TO_GESTURE_MAP['A#4'][0] == 'Rock On'
