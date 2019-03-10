#!/usr/bin/env python
# -*- encoding: utf-8 -*-

__author__ = 'andyguo'

import re
from collections import namedtuple
PatternHanlder = namedtuple('PatternHanlder', 're, get_padding, format')

# 用于判断frame 的正则表达式
FRAME_REGEX = re.compile(r'(?<![\dvV])(\d{2,})$')
PATTERN_HANDLERS = {
    '%': PatternHanlder(
            re=re.compile(r'\%(\d*)d'),
            get_padding=lambda m: int(m.group(1) or 1),
            format='%0{}d'.format,
    ),
    '#': PatternHanlder(
        re=re.compile(r'#+'),
        get_padding=lambda m: len(m.group(0)),
        format=lambda l: '#' * l,
    ),
    '$': PatternHanlder(
        re=re.compile(r'\$F(\d*)'),
        get_padding=lambda m: int(m.group(1) or 1),
        format='$F{}'.format,
    ),
    '*': PatternHanlder(
        re=re.compile(r'\*'),
        get_padding=lambda m: None,
        format='*'.format,
    ),
}

PATTERN_REGEX_MAP = {
    '%': re.compile(r'\%(\d*)d'),
    '#': re.compile(r'#+'),
    '$': re.compile(r'\$F(\d*)'),
    '*': re.compile(r'\*'),
}
PADDING_LENTH_GETTER = {
    '%': lambda m: int(m.group(1)),
    '#': lambda m: len(m.group(0)),
    '$': lambda m: int(m.group(1)),
    '*': lambda m: None,
}
PADDING_LENTH_GETTER = {
    '%': lambda m: int(m.group(1)),
    '#': lambda m: len(m.group(0)),
    '$': lambda m: int(m.group(1)),
    '*': lambda m: None,
}

# 用于小写化 windows 盘符的正则表达式
WIN32_DRIVE_REGEX = re.compile(r'^(\w:).*')
# NUC regex
UNC_REGEX = re.compile(r'^(\\\\.*?)/+.*')

VERSION_REGEX = re.compile(r'.*([vV]\d+).*')

EXT_SINGLE_MEDIA = {'.mov': {},
                    '.mp4': {},
                    '.avi': {},
                    '.r3d': {},
                    }

SCAN_IGNORE = {'start': ('.', '..', 'Thumb'),
               'end'  : ('.tmp')}
ADDITIONAL_EXTS = ['.bgeo.sc']

NETWORK_FILE_SYSTEM = ('nfs', 'smbfs', 'remote', 'afp', 'ftp', 'snfs')
