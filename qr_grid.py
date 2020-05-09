# -*- coding: utf-8 -*-

"""QR and MaskEvaluation classes.

This module creates and saves QR image.
What it needs is a message in form of a bitstring
and Error correction level that's been used.
EC level takes values from 0 to 3, which stays for:
0 Level L (Low) 	    ~7% of codewords can be restored.
1 Level M (Medium) 	    ~15% of codewords can be restored.
2 Level Q (Quartile) 	~25% of codewords can be restored.
3 Level H (High) 	    ~30% of codewords can be restored.

Step by step instruction to create QR code image:

# qr = QR(version)
# qr.insert_message(bitstring_message)
# qr.apply_mask(eclvl, mask)
# qr.add_quiet_zone(margin)
# qr.save_qr(fp, scale, show_result)

The grid is constructed as numpy 2D array.
"""


from os.path import abspath
from itertools import product

import numpy as np
from PIL import Image
from CONSTANTS import ALIGNMENT_COORDS, FORMAT_INFO, VERSION_INFO


# pylint: disable=invalid-name

# Prepared patterns ready for later insertion into QR grid.
FINDER = np.zeros((7, 7))
FINDER[1:6, 1] = FINDER[1:6, 5] = FINDER[1, 1:6] = FINDER[5, 1:6] = 255

SEPARATOR = np.ones((1, 8)) * 255

ALIGNMENT_SQUARE = np.zeros((5, 5))
ALIGNMENT_SQUARE[1:4, 1:4] = 255
ALIGNMENT_SQUARE[2, 2] = 0


def get_alignment_coords(qr_version):
    """Get alignment patterns coordinates according to qr code version.
    Return cross product of given positions for all center module locations."""
    if qr_version == 1:
        return ()

    positions = ALIGNMENT_COORDS[qr_version]
    return tuple(product(positions, positions))


class QR(object):
    """Creates QR code, inserts a message and optionally saves it at a given location.
    Public methods:
        apply_mask(ec_lvl, mask=None)
        insert_message(message)
        add_quiet_zone(margin=4)
        save_qr(fp, scale)
    """
    _masks = {0: lambda r, c: (r + c) % 2 == 0,
              1: lambda r, c: r % 2 == 0,
              2: lambda r, c: c % 3 == 0,
              3: lambda r, c: (r + c) % 3 == 0,
              4: lambda r, c: (r // 2 + c // 3) % 2 == 0,
              5: lambda r, c: (r * c) % 2 + (r * c) % 3 == 0,
              6: lambda r, c: ((r * c) % 2 + (r * c) % 3) % 2 == 0,
              7: lambda r, c: ((r + c) % 2 + (r * c) % 3) % 2 == 0}

    def __init__(self, version):
        self.ver = version
        self.qr_size = 17 + version * 4

        # Create qr grid with all modules equaled to 100. Such modules will be
        # replaced by appropriate patterns systematically.
        self.qr = np.zeros((self.qr_size, self.qr_size), dtype=np.uint8) + 100
        self._insert_patterns()

        self._mask_reference = self.qr.copy()
        self.mask_num = None

    def _add_finders(self):
        """Inserts 3 finder patterns."""
        self.qr[0:7, 0:7] = FINDER  # top left
        self.qr[-7:, 0:7] = FINDER  # bottom left
        self.qr[0:7, -7:] = FINDER  # top right

    def _add_separators(self):
        """Inserts 6 separator patterns, placed around finder squares."""
        self.qr[7, :8] = SEPARATOR
        self.qr[:8, 7] = SEPARATOR
        self.qr[-8, :8] = SEPARATOR
        self.qr[-8:, 7] = SEPARATOR
        self.qr[:8, -8] = SEPARATOR
        self.qr[7, -8:] = SEPARATOR

    def _add_alignment_patterns(self):
        """Inserts alignment patterns, according to their specified locations
        returned by get_alignment_coords() function. Such a pattern can only be
        drawn if all targeted blocks are equal to 100 to make sure it won't
        overlap the finder patterns nor separators.
        """
        coords = get_alignment_coords(self.ver)
        for x, y in coords:
            if np.any([self.qr[x - 2:x + 3, y - 2:y + 3] != 100]):
                continue
            self.qr[x - 2:x + 3, y - 2:y + 3] = ALIGNMENT_SQUARE

    def _add_timing_pattern(self):
        """Inserts 2 timing patterns. Continuous black and white pixels line
        between finder patterns.
        """
        timing_pat_len = self.qr_size - 16
        timing_line = [0 if c % 2 == 0 else 255 for c in range(timing_pat_len)]
        self.qr[6, 8:-8] = timing_line  # horizontal
        self.qr[8:-8, 6] = timing_line  # vertical

    def _add_reserved_areas(self):
        """Inserts dark module and reserved format info areas. All such blocks
        will have 155 value until they're replaced by final format, version info.
        """
        # Dark module
        self.qr[-8, 8] = 0

        # Reserved format info areas
        self.qr[:9, 8][self.qr[:9, 8] == 100] = 155
        self.qr[8, :9][self.qr[8, :9] == 100] = 155
        self.qr[-7:, 8][self.qr[-7:, 8] == 100] = 155
        self.qr[8, -8:][self.qr[8, -8:] == 100] = 155

    def _add_version_info(self):
        """For qr codes versions bigger than 6, version info has to be added."""
        if self.ver >= 7:
            ver_info = map(int, format(VERSION_INFO[self.ver], '018b'))
            ver_info = np.array([255 if i == 0 else 0 for i in ver_info][::-1]).reshape(6, 3)
            self.qr[:6, -11:-8] = ver_info  # top right
            self.qr[-11:-8, :6] = ver_info.transpose()  # bottom left

    def _insert_patterns(self):
        """Inserts all above patterns."""
        self._add_finders()
        self._add_separators()
        self._add_alignment_patterns()
        self._add_timing_pattern()
        self._add_reserved_areas()
        self._add_version_info()

    @staticmethod
    def _add_format_info(qr, mask, ec_num):
        """Informs about error correction level and mask number applied."""
        format_info = map(int, format(FORMAT_INFO[ec_num][mask], '015b'))
        format_info = [255 if i == 0 else 0 for i in format_info]
        qr[:9, 8][qr[:9, 8] == 155] = format_info[7:][::-1]  # top left vertical
        qr[8, :8][qr[8, :8] == 155] = format_info[:7]        # top left horizontal
        qr[-7:, 8][qr[-7:, 8] == 155] = format_info[:7][::-1]  # bottom left vertical
        qr[8, -8:][qr[8, -8:] == 155] = format_info[7:]      # top right horizontal

    def _add_mask(self, qr, qr_ref, mask, ec_lvl):
        """qr_ref is a grid copy without data and error correction.
        Empty modules have values of 100 and indicate valid fields for mask application.
        qr is a complete grid with a message.
        """
        qr_masked = qr.copy()
        self._add_format_info(qr_masked, mask, ec_lvl)
        size = qr_masked.shape[0]
        for row in range(size):
            for col in range(size):
                if self._masks[mask](row, col) and qr_ref[row, col] == 100:
                    qr_masked[row, col] ^= 255  # switch between 0 and 255

        penalty = MaskEvaluation.get_penalty(qr_masked)
        return penalty, qr_masked

    def _apply_best(self, qr, qr_ref, eclvl):
        """Check all masks and choose the best one."""
        min_penalty = float('inf')
        masked_qr, mask_num = None, None
        for m in self._masks:
            score, qr_square = self._add_mask(qr, qr_ref, m, eclvl)
            if score < min_penalty:
                min_penalty = score
                masked_qr = qr_square
                mask_num = m

        print("[+] Best mask has been applied: [{}]. With {} penalty".format(mask_num, min_penalty))
        return masked_qr, mask_num

    def apply_mask(self, ec_lvl, mask=None):
        """If masks is not specified, it will be chosen automatically."""
        if mask is None:
            self.qr, self.mask_num = self._apply_best(self.qr, self._mask_reference, ec_lvl)
        else:
            assert 0 <= mask <= 7, "Mask number incorrect {}. " \
                                   "Number between 0 and 7 expected.".format(mask)
            self.mask_num = mask
            _, self.qr = self._add_mask(self.qr, self._mask_reference, mask, ec_lvl)

    def insert_message(self, mes):
        """INPUT: bitstring message.
        Data filling starts from 2 bottom right columns and goes upwards until
        the top is reached. Next, it goes downwards by 2 columns until it reaches
        the bottom of a grid and so on. The shape of filling pattern takes a form
        of a snake-like line.

        Only unused grid modules are used, which are those equal to 100. Also, the column
        with vertical timing pattern is temporarily cut out, since it's not involved in
        insertion process, but could affect proper run of a "snake" line.
        It's brought back after the message is inserted.
        """
        # Remove vertical timing pattern column.
        timing_col = self.qr[:, 6]
        self.qr = np.delete(self.qr, 6, axis=1)

        # Replace bits with black and white 8-bit color representation
        # and reverse the list to get right values while popping from the end.
        mes = [255 if char == '0' else 0 for char in mes][::-1]
        go_upwards = 1
        for idx in range(2, self.qr_size + 1, 2):
            if idx == 2:
                col_pair = self.qr[:, -idx:][::-1]
            else:
                # Change of direction is done by reversing a slice.
                if go_upwards == 1:
                    col_pair = self.qr[:, -idx:-idx + 2][::-1]
                else:
                    col_pair = self.qr[:, -idx:-idx + 2]

            height, width = col_pair.shape
            for r in range(height):
                for c in range(1, 1 + width):
                    if col_pair[r, -c] == 100:
                        col_pair[r, -c] = mes.pop()
            go_upwards ^= 1

        # Bring back vertical timing pattern column.
        self.qr = np.insert(self.qr, 6, timing_col, axis=1)

    def add_quiet_zone(self, margin=4):
        """Adds the margin to each side of the qr code. It's done by inserting
        qr code into extended grid (with margin included).
        """
        idx_end = self.qr_size + margin
        new_dimension = self.qr_size + margin * 2
        quiet_zone = np.ones([new_dimension, new_dimension]) * 255
        quiet_zone[margin:idx_end, margin:idx_end] = self.qr
        self.qr_size = new_dimension
        self.qr = quiet_zone

    def save_qr(self, fp=None, scale=10, show_result=True):
        """Saves qr code under the given filename.
        The format to use is determined from the filename extension.
        Preview QR image is shown, no matter if saved or not.
        """
        i = Image.new('L', (self.qr_size, self.qr_size))
        i.putdata(self.qr.flatten())

        new_dim = self.qr_size * scale
        i = i.resize(size=(new_dim, new_dim))
        if fp:
            i.save(fp)
            print("QR saved at: {}".format(abspath(fp)))
        if show_result:
            i.show()


class MaskEvaluation:
    """Penalty score is evaluated as a sum of penalties from checking 4 conditions."""

    @staticmethod
    def cond1(qr):
        """Check each row and each column one-by-one. If there are five
        consecutive modules of the same color, add 3 to the penalty.
        If there are more modules of the same color after the first five,
        add 1 for each additional module of the same color.
        """
        penalty = 0
        for qr_grid in (qr, qr.transpose()):
            for row in qr_grid:
                module = None
                consecutive = 0
                for val in row:
                    if val == module:
                        consecutive += 1
                    else:
                        module = val
                        consecutive = 1

                    if consecutive < 5:
                        continue
                    elif consecutive == 5:
                        penalty += 3
                    else:
                        penalty += 1
        return penalty

    @staticmethod
    def cond2(qr):
        """add 3 to the penalty score for every 2x2 block of the same color in the QR code"""
        dim = qr.shape[0]
        penalty = 0
        for r in range(dim - 1):
            for c in range(dim - 1):
                if qr[r, c] == qr[r+1, c] == qr[r, c+1] == qr[r+1, c+1]:
                    penalty += 3
        return penalty

    @staticmethod
    def cond3(qr):
        """Look for a given pattern, when it's found add 40 to the penalty score."""
        pat = np.array([0, 255, 0, 0, 0, 255, 0, 255, 255, 255, 255])  # pattern
        pat_rev = np.array([255, 255, 255, 255, 0, 255, 0, 0, 0, 255, 0])  # pattern reversed
        pat_len, dim = len(pat), qr.shape[0]

        penalty = 0
        for q in (qr, qr.transpose()):
            for r in range(dim):
                for c in range(dim - pat_len + 1):
                    sliced = q[r, c:c + pat_len]
                    if not (sliced != pat).any() or not (sliced != pat_rev).any():
                        penalty += 40
        return penalty

    @staticmethod
    def cond4(qr):
        """Penalty based on ratio of black to white modules."""
        total = qr.size
        blacks = np.count_nonzero([qr == 0])
        blacks_ratio = blacks / total * 100

        # Determine the previous and next multiple of five of the ratio. Subtract 50
        # from each of them and divide the absolute result by 5.
        # Finally, choose smaller number and multiply by 10.
        p = abs(blacks_ratio // 5 * 5 - 50) / 5
        n = abs(np.ceil(blacks_ratio / 5) * 5 - 50) / 5
        return min(p, n) * 10

    @classmethod
    def get_penalty(cls, arr):
        """Sum of penalties for a given mask test."""
        return sum(f(arr) for f in (cls.cond1, cls.cond2, cls.cond3, cls.cond4))
