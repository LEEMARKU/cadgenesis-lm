import sys
sys.path.insert(0, 'D:/Gen-AI CAD_LLM/src')

import cadgenesis.datasets.cad_program_synth as cps

# Append new templates to the list
new_templates = [
    {
        'name': 'counterbore hole',
        'prompt': 'a counterbore hole with {d}mm diameter and {d2}mm counterbore depth',
        'tokens': ['COUNTERBORE', '@d', 'EXTRUDE', '@d2', 'HOLE'],
    },
    {
        'name': 'slot',
        'prompt': 'a slot {l}mm long and {w}mm wide',
        'tokens': ['SLOT', '@l', 'EXTRUDE', '@w', 'RECT'],
    },
    {
        'name': 'fillet',
        'prompt': 'a fillet {r}mm radius on a {s}mm side',
        'tokens': ['FILLET', '@r', 'EDGE', '@s'],
    },
    {
        'name': 'tolerance stack',
        'prompt': 'a tolerance stack of {n} parts each {t}mm',
        'tokens': ['PART', '@n', '@t', 'EXTRUDE', 'NUM_10'],
    },
    {
        'name': 'mating dowel',
        'prompt': 'a dowel joint with {d}mm diameter peg and {h}mm hole',
        'tokens': ['DOWEL', '@d', 'HOLE', '@h', 'EXTRUDE'],
    },
    {
        'name': 'two-part assembly',
        'prompt': 'a {w}mm x {h}mm base with {d}mm peg on top',
        'tokens': ['BASE', '@w', '@h', 'PEG', '@d', 'EXTRUDE'],
    },
    {
        'name': 'external thread',
        'prompt': 'an external thread {d}mm diameter with {p}mm pitch',
        'tokens': ['THREAD', '@d', '@p', 'CYLINDER'],
    },
    {
        'name': 'counterbore bolt hole',
        'prompt': 'a counterbore bolt hole {d}mm diameter with {c}mm countersink',
        'tokens': ['COUNTERBORE', '@d', '@c', 'HOLE', 'BOLT'],
    },
    {
        'name': 'weight calculation',
        'prompt': 'calculate the weight of a {w}mm x {h}mm x {d}mm steel block',
        'tokens': ['STEEL', '@w', '@h', '@d', 'WEIGHT', 'VOLUME'],
    },
    {
        'name': 'clearance fit',
        'prompt': 'a {d}mm shaft in a {H}mm hole with {t}mm clearance',
        'tokens': ['SHAFT', '@d', 'HOLE', '@H', 'CLEARANCE', '@t'],
    },
    {
        'name': 'complete bracket',
        'prompt': 'a complete mounting bracket for a {w}mm panel with {h}mm height and {d}mm depth',
        'tokens': ['BRACKET', '@w', '@h', '@d', 'MOUNT', 'EXTRUDE', 'SLOT'],
    },
    {
        'name': 'complex fixture',
        'prompt': 'a complex fixture with {n} holes {d}mm diameter and {s}mm spacing',
        'tokens': ['FIXTURE', '@n', '@d', 'SPACING', 'HOLE', 'PATTERN', 'EXTRUDE'],
    },
]

cps._TEMPLATES.extend(new_templates)
print(f'Added {len(new_templates)} new templates')
print(f'Total templates now: {len(cps._TEMPLATES)}')

for t in new_templates[:3]:
    print(f'  {t["name"]}: {t["prompt"]}')
    print(f'    tokens: {t["tokens"]}')