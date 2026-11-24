"""Self-contained 8-bar MusicXML sample score for runnable `hw-score-sim` examples.

Includes treble clef, key signature (D major / 2 sharps), accidentals (sharp, flat,
natural), 8th/16th beamed groups, unbeamed flagged notes, chords, rests, dynamics,
hairpins, ties, slurs, and ledger lines so every example runs out-of-the-box with
zero external files.
"""

from __future__ import annotations

from pathlib import Path

SAMPLE_SCORE_MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <work>
    <work-title>Etude in D Major (hw-score-sim Sample)</work-title>
  </work>
  <part-list>
    <score-part id="P1">
      <part-name>Violin / Piano RH</part-name>
    </score-part>
  </part-list>
  <part id="P1">
    <!-- Measure 1: Clef, key sig (D major), tempo, mf, unbeamed flags & rest -->
    <measure number="1">
      <attributes>
        <divisions>4</divisions>
        <key><fifths>2</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <direction placement="above">
        <direction-type><words>Allegro con brio</words></direction-type>
      </direction>
      <direction placement="below">
        <direction-type><dynamics><mf/></dynamics></direction-type>
      </direction>
      <note>
        <pitch><step>D</step><octave>4</octave></pitch>
        <duration>4</duration><type>quarter</type><stem>up</stem>
        <notations><slur type="start" number="1"/></notations>
      </note>
      <note>
        <pitch><step>F</step><alter>1</alter><octave>4</octave></pitch>
        <duration>2</duration><type>eighth</type><stem>up</stem>
      </note>
      <note>
        <rest/><duration>2</duration><type>eighth</type>
      </note>
      <note>
        <pitch><step>A</step><octave>4</octave></pitch>
        <duration>1</duration><type>16th</type><stem>up</stem>
      </note>
      <note>
        <rest/><duration>3</duration><type>eighth</type><dot/>
      </note>
      <note>
        <pitch><step>D</step><octave>5</octave></pitch>
        <duration>4</duration><type>quarter</type><stem>down</stem>
        <notations><slur type="stop" number="1"/></notations>
      </note>
    </measure>

    <!-- Measure 2: Beamed 8ths + beamed 16ths + accidentals -->
    <measure number="2">
      <direction placement="below">
        <direction-type><wedge type="crescendo"/></direction-type>
      </direction>
      <note>
        <pitch><step>E</step><octave>5</octave></pitch>
        <duration>2</duration><type>eighth</type><stem>down</stem>
        <beam number="1">begin</beam>
      </note>
      <note>
        <pitch><step>D</step><octave>5</octave></pitch>
        <duration>2</duration><type>eighth</type><stem>down</stem>
        <beam number="1">end</beam>
      </note>
      <note>
        <pitch><step>C</step><alter>0</alter><octave>5</octave></pitch>
        <duration>1</duration><type>16th</type><accidental>natural</accidental><stem>down</stem>
        <beam number="1">begin</beam><beam number="2">begin</beam>
      </note>
      <note>
        <pitch><step>B</step><octave>4</octave></pitch>
        <duration>1</duration><type>16th</type><stem>down</stem>
        <beam number="1">continue</beam><beam number="2">continue</beam>
      </note>
      <note>
        <pitch><step>A</step><octave>4</octave></pitch>
        <duration>1</duration><type>16th</type><stem>down</stem>
        <beam number="1">continue</beam><beam number="2">continue</beam>
      </note>
      <note>
        <pitch><step>G</step><alter>1</alter><octave>4</octave></pitch>
        <duration>1</duration><type>16th</type><accidental>sharp</accidental><stem>down</stem>
        <beam number="1">end</beam><beam number="2">end</beam>
      </note>
      <note>
        <pitch><step>A</step><octave>4</octave></pitch>
        <duration>4</duration><type>quarter</type><stem>up</stem>
      </note>
      <note>
        <pitch><step>B</step><alter>-1</alter><octave>4</octave></pitch>
        <duration>4</duration><type>quarter</type><accidental>flat</accidental><stem>down</stem>
        <tie type="start"/>
        <notations><tied type="start"/></notations>
      </note>
      <direction placement="below">
        <direction-type><wedge type="stop"/></direction-type>
      </direction>
    </measure>

    <!-- Measure 3: Tie resolution, chords, and forte dynamic -->
    <measure number="3">
      <direction placement="below">
        <direction-type><dynamics><f/></dynamics></direction-type>
      </direction>
      <note>
        <pitch><step>B</step><alter>-1</alter><octave>4</octave></pitch>
        <duration>4</duration><type>quarter</type><stem>down</stem>
        <tie type="stop"/>
        <notations><tied type="stop"/></notations>
      </note>
      <note>
        <pitch><step>A</step><octave>4</octave></pitch>
        <duration>4</duration><type>quarter</type><stem>up</stem>
      </note>
      <note>
        <chord/>
        <pitch><step>F</step><alter>1</alter><octave>4</octave></pitch>
        <duration>4</duration><type>quarter</type><stem>up</stem>
      </note>
      <note>
        <chord/>
        <pitch><step>D</step><octave>4</octave></pitch>
        <duration>4</duration><type>quarter</type><stem>up</stem>
      </note>
      <note>
        <pitch><step>G</step><octave>4</octave></pitch>
        <duration>4</duration><type>quarter</type><stem>up</stem>
      </note>
      <note>
        <chord/>
        <pitch><step>B</step><octave>4</octave></pitch>
        <duration>4</duration><type>quarter</type><stem>up</stem>
      </note>
      <note>
        <rest/><duration>4</duration><type>quarter</type>
      </note>
    </measure>

    <!-- Measure 4: Slurred lyrical phrase with ledger lines -->
    <measure number="4">
      <direction placement="below">
        <direction-type><dynamics><p/></dynamics></direction-type>
      </direction>
      <note>
        <pitch><step>A</step><octave>3</octave></pitch>
        <duration>4</duration><type>quarter</type><stem>up</stem>
        <notations><slur type="start" number="1"/></notations>
      </note>
      <note>
        <pitch><step>D</step><octave>4</octave></pitch>
        <duration>2</duration><type>eighth</type><stem>up</stem>
        <beam number="1">begin</beam>
      </note>
      <note>
        <pitch><step>F</step><alter>1</alter><octave>4</octave></pitch>
        <duration>2</duration><type>eighth</type><stem>up</stem>
        <beam number="1">end</beam>
      </note>
      <note>
        <pitch><step>A</step><octave>4</octave></pitch>
        <duration>4</duration><type>quarter</type><stem>up</stem>
      </note>
      <note>
        <pitch><step>A</step><octave>5</octave></pitch>
        <duration>4</duration><type>quarter</type><stem>down</stem>
        <notations><slur type="stop" number="1"/></notations>
      </note>
    </measure>

    <!-- Measure 5: Dotted rhythm + unbeamed downward flag + chords -->
    <measure number="5">
      <note>
        <pitch><step>B</step><octave>5</octave></pitch>
        <duration>6</duration><type>quarter</type><dot/><stem>down</stem>
      </note>
      <note>
        <pitch><step>A</step><octave>5</octave></pitch>
        <duration>2</duration><type>eighth</type><stem>down</stem>
      </note>
      <note>
        <pitch><step>G</step><octave>5</octave></pitch>
        <duration>4</duration><type>quarter</type><stem>down</stem>
      </note>
      <note>
        <chord/>
        <pitch><step>E</step><octave>5</octave></pitch>
        <duration>4</duration><type>quarter</type><stem>down</stem>
      </note>
      <note>
        <pitch><step>F</step><alter>1</alter><octave>5</octave></pitch>
        <duration>4</duration><type>quarter</type><stem>down</stem>
      </note>
      <note>
        <chord/>
        <pitch><step>D</step><octave>5</octave></pitch>
        <duration>4</duration><type>quarter</type><stem>down</stem>
      </note>
    </measure>

    <!-- Measure 6: Rapid 16th-note run + dim hairpin -->
    <measure number="6">
      <direction placement="below">
        <direction-type><wedge type="diminuendo"/></direction-type>
      </direction>
      <note>
        <pitch><step>E</step><octave>5</octave></pitch>
        <duration>1</duration><type>16th</type><stem>down</stem>
        <beam number="1">begin</beam><beam number="2">begin</beam>
      </note>
      <note>
        <pitch><step>D</step><octave>5</octave></pitch>
        <duration>1</duration><type>16th</type><stem>down</stem>
        <beam number="1">continue</beam><beam number="2">continue</beam>
      </note>
      <note>
        <pitch><step>C</step><alter>1</alter><octave>5</octave></pitch>
        <duration>1</duration><type>16th</type><stem>down</stem>
        <beam number="1">continue</beam><beam number="2">continue</beam>
      </note>
      <note>
        <pitch><step>B</step><octave>4</octave></pitch>
        <duration>1</duration><type>16th</type><stem>down</stem>
        <beam number="1">end</beam><beam number="2">end</beam>
      </note>
      <note>
        <pitch><step>A</step><octave>4</octave></pitch>
        <duration>2</duration><type>eighth</type><stem>up</stem>
        <beam number="1">begin</beam>
      </note>
      <note>
        <pitch><step>G</step><octave>4</octave></pitch>
        <duration>2</duration><type>eighth</type><stem>up</stem>
        <beam number="1">end</beam>
      </note>
      <note>
        <pitch><step>F</step><alter>1</alter><octave>4</octave></pitch>
        <duration>4</duration><type>quarter</type><stem>up</stem>
      </note>
      <note>
        <pitch><step>E</step><octave>4</octave></pitch>
        <duration>4</duration><type>quarter</type><stem>up</stem>
      </note>
      <direction placement="below">
        <direction-type><wedge type="stop"/></direction-type>
      </direction>
    </measure>

    <!-- Measure 7: Syncopated tie + half rest -->
    <measure number="7">
      <note>
        <pitch><step>F</step><alter>1</alter><octave>4</octave></pitch>
        <duration>2</duration><type>eighth</type><stem>up</stem>
      </note>
      <note>
        <pitch><step>G</step><octave>4</octave></pitch>
        <duration>2</duration><type>eighth</type><stem>up</stem>
        <tie type="start"/>
        <notations><tied type="start"/></notations>
      </note>
      <note>
        <pitch><step>G</step><octave>4</octave></pitch>
        <duration>4</duration><type>quarter</type><stem>up</stem>
        <tie type="stop"/>
        <notations><tied type="stop"/></notations>
      </note>
      <note>
        <rest/><duration>8</duration><type>half</type>
      </note>
    </measure>

    <!-- Measure 8: Cadential chord + final barline -->
    <measure number="8">
      <direction placement="below">
        <direction-type><dynamics><ff/></dynamics></direction-type>
      </direction>
      <note>
        <pitch><step>A</step><octave>4</octave></pitch>
        <duration>4</duration><type>quarter</type><stem>up</stem>
      </note>
      <note>
        <chord/>
        <pitch><step>C</step><alter>1</alter><octave>5</octave></pitch>
        <duration>4</duration><type>quarter</type><stem>up</stem>
      </note>
      <note>
        <pitch><step>D</step><octave>4</octave></pitch>
        <duration>12</duration><type>half</type><dot/><stem>up</stem>
      </note>
      <note>
        <chord/>
        <pitch><step>F</step><alter>1</alter><octave>4</octave></pitch>
        <duration>12</duration><type>half</type><dot/><stem>up</stem>
      </note>
      <note>
        <chord/>
        <pitch><step>A</step><octave>4</octave></pitch>
        <duration>12</duration><type>half</type><dot/><stem>up</stem>
      </note>
      <note>
        <chord/>
        <pitch><step>D</step><octave>5</octave></pitch>
        <duration>12</duration><type>half</type><dot/><stem>up</stem>
      </note>
      <barline location="right">
        <bar-style>light-heavy</bar-style>
      </barline>
    </measure>
  </part>
</score-partwise>
"""


def get_sample_musicxml() -> str:
    """Return the bundled 8-bar D-major sample MusicXML string."""
    return SAMPLE_SCORE_MUSICXML


def write_sample_score(path: str | Path) -> Path:
    """Write the bundled 8-bar sample MusicXML score to ``path`` and return it."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(SAMPLE_SCORE_MUSICXML, encoding="utf-8")
    return dest
