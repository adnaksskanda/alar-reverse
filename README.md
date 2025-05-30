# Alar: Kannada - English dictionary (reverse lookup of English definitions)
 The Alar dictionary corpus is a free and open source Kannada - English dictionary created by V. Krishna. It has over 150,000 Kannada entries and over 240,000 English definitions. [Read the full story here](https://zerodha.tech/blog/alar-the-making-of-an-open-source-dictionary).

 I decided to hack together a lookup of the English definitions, to enable searching of Kannada words with a given English search prompt. It is pretty makeshift, please pardon the shoddy design (used a LLM to code most of it). However now we have a tool that can search through the alar.ink corpus if we want to find a Kannada word that matches an English word we know. This should only serve as a temporary tool until V. Krishna sir is finished with his English-Kannada dictionary project.

 The data is licensed under ODC-OdBL (c) V. Krishna.

## Usage
The corpus is available as a searchable dictionary on [https://alar.ink](https://alar.ink).

This repository is a website that simply searches through the English definitions in alar.ink and pulls up Kannada matches.

## Format
The corpus is a single YAML file. Each entry is in the following format.

```yaml
- id: 59 # Orignal ID of the entry
  head: ಅ # Alphabet / first letter of the entry word.
  entry: ಅಂಕಪರದೆ # Entry Word.
  phone: aŋka parade # Phonetic notation.
  origin: ""
  info: ""
  defs:
  - id: 335427
    entry: the curtain used to pull down at the end of a scene or act, in a play.
    type: noun
  - id: 211623
    entry: (fig.) the end of or an action bringing an end to, an event or an occasion;
    type: noun
  - id: 336691
    entry: ಅಂಕಪರದೆಯೇಳು aŋkaparadeyēḷu (the performance of a dramatic act) to start (as on the stage).
    type: noun
  - id: 237657
    entry: (fig.) (a new phase of action, style, etc.) to commence; ಅಂಕಪರದೆಬೀಳು aŋkaparade bīḷu to come to an end; 2. to cause to end.
    type: noun
```

## License
ODC-ODbL
