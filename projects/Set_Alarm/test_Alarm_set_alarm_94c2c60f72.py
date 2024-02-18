# ********RoostGPT********
"""
Test generated by RoostGPT for test MiniProjects using AI Type Azure Open AI and AI Model roost-gpt4-32k

1. Scenario: Provide a valid time for the alarm in the format (HH:MM). 
   - Expected: The function should set the alarm for the given time.

2. Scenario: Provide an invalid time for the alarm (not in the format HH:MM).
   - Expected: It should show an error "Time format invalid! Please try again!"

3. Scenario: Check the behavior when a music directory is empty.
   - Expected: The function should return an error "No music in the musics folder! Please add music first!"

4. Scenario: Check what happens when only one music file is in the music folder.
   - Expected: The function should show a message "Alarm music has been set default" with the only music file's name.

5. Scenario: Provide a valid index number when there are more than one music files.
   - Expected: The function should show a message "Alarm music has been set" with the selected music file's name.

6. Scenario: Provide an invalid index number (negative/ zero/ beyond the range of available music files) when there are more than one music files.
   - Expected: It should show an error "Invalid Index! Please try again!"

7. Scenario: Check the behavior when the input time is the current system time. 
   - Expected: The alarm should be immediately activated with the selected music file.

8. Scenario: Check the behavior when the current time passed the input time.
   - Expected: The alarm should be immediately activated with the selected music file.

9. Scenario: Check the behavior when the current time has not yet reached input time.
   - Expected: The function should continuously check for the time and wait until the system time equals the input time and then, the alarm should be activated with the selected music file.
"""

# ********RoostGPT********
import pytest
import alarm
import os
from unittest.mock import patch, MagicMock
import datetime
import subprocess

# Mocking the subprocess.run method
subprocess.run = MagicMock()

def test_alarm_valid_time(monkeypatch):
    # Scenario 1: Provide a valid time for the alarm in the format (HH:MM).
    monkeypatch.setattr('builtins.input', lambda _: "12:30")
    alarm.set_alarm()
    subprocess.run.assert_called_once()

def test_alarm_invalid_time(monkeypatch):
    # Scenario 2: Provide an invalid time for the alarm (not in the format HH:MM).
    monkeypatch.setattr('builtins.input', lambda _: "1230")
    with pytest.raises(SystemExit) as ex:
        alarm.set_alarm()
    assert 'Time format invalid! Please try again!\n' in str(ex.value)

def test_alarm_no_music_files(monkeypatch):
    # Scenario 3: Check the behavior when a music directory is empty.
    monkeypatch.setattr('builtins.input', lambda _: "12:30")
    monkeypatch.setattr('os.listdir', lambda _: [])
    with pytest.raises(SystemExit) as ex:
        alarm.set_alarm()
    assert 'No music in the musics folder! Please add music first!\n' in str(ex.value)

def test_alarm_one_music_file(monkeypatch, capsys):
    # Scenario 4: Check what happens when only one music file is in the music folder.
    monkeypatch.setattr('builtins.input', lambda _: "12:30")
    monkeypatch.setattr('os.listdir', lambda _: ['my_music.mp3'])
    alarm.set_alarm()
    out, err = capsys.readouterr()
    assert 'Alarm music has been set default --> My Music' in out

def test_alarm_valid_music_index(monkeypatch, capsys):
    # Scenario 5: Provide a valid index number when there are more than one music files.
    monkeypatch.setattr('builtins.input', lambda x: "12:30" if 'alarm' in x else "1")
    monkeypatch.setattr('os.listdir', lambda _: ['music1.mp3', 'music2.mp3'])
    alarm.set_alarm()
    out, err = capsys.readouterr()
    assert 'Alarm music has been set --> Music1' in out

def test_alarm_invalid_music_index(monkeypatch):
    # Scenario 6: Provide an invalid index number (negative/ zero/ beyond the range of available music files) when there are more than one music files.
    monkeypatch.setattr('builtins.input', lambda x: "12:30" if 'alarm' in x else "3")
    monkeypatch.setattr('os.listdir', lambda _: ['music1.mp3', 'music2.mp3'])
    with pytest.raises(SystemExit) as ex:
        alarm.set_alarm()
    assert 'Invalid Index! Please try again!\n' in str(ex.value)

def test_alarm_current_system_time(monkeypatch):
    # Scenario 7: Check the behavior when the input time is the current system time. 
    current_time = datetime.datetime.now().time()
    monkeypatch.setattr('builtins.input', lambda _: f"{current_time.hour}:{current_time.minute}")
    monkeypatch.setattr('os.listdir', lambda _: ['music.mp3'])
    alarm.set_alarm()
    subprocess.run.assert_called_once()

def test_alarm_past_system_time(monkeypatch):
    # Scenario 8: Check the behavior when the current time passed the input time.
    past_time = (datetime.datetime.now() - datetime.timedelta(minutes=2)).time()
    monkeypatch.setattr('builtins.input', lambda _: f"{past_time.hour}:{past_time.minute}")
    monkeypatch.setattr('os.listdir', lambda _: ['music.mp3'])
    alarm.set_alarm()
    subprocess.run.assert_called_once()

def test_alarm_future_system_time(monkeypatch, capsys):
    # Scenario 9: Check the behavior when the current time has not yet reached input time.
    future_time = (datetime.datetime.now() + datetime.timedelta(minutes=2)).time()
    monkeypatch.setattr('builtins.input', lambda _: f"{future_time.hour}:{future_time.minute}")
    monkeypatch.setattr('os.listdir', lambda _: ['music.mp3'])
    alarm.set_alarm()
    out, err = capsys.readouterr()
    assert f"Alarm has been set successfully for {future_time.hour}:{future_time.minute}!" in out
