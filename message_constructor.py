# -*- coding: utf-8 -*-

"""RSEncoder, ModeEncoder and Message classes.

This module handles message construction for QR code.
It encodes message with the most appropriate encoding (numeric, alphanumeric, bytes).
Next, mode indicator, character count indicator, terminator and pad bytes are joined
together with the encoded message. After that Reed Solomon error correction is applied.

Step by step instruction to get the final message:

# m = Message(text_message, version=version, ec_level=eclvl)
# ecc_num = m.get_ecc_num()
# version = m.get_version()
# rs = RSEncoder(ecc_num)
#
# mes_blocks = m.get_segments()
# ecc_blocks = rs.get_ecc_from_blocks(mes_blocks)
#
# full_mes = rs.interleave_blocks(mes_blocks) + rs.interleave_blocks(ecc_blocks)
# full_mes = m.codewords_to_bitstr(full_mes)
# full_mes += m.get_remainder_bits()

Remarks:
Kanji is handled by UTF-8 at the moment. However, straight kanji encoding
will save fair number of bytes, thus longer messages become available.

To get some detailed view about how qr code is constructed,
I recommend visiting: https://www.thonky.com/qr-code-tutorial

For Reed Solomon error correction:
http://downloads.bbc.co.uk/rd/pubs/whp/whp-pdf-files/WHP031.pdf
"""


from CONSTANTS import MODE_INDICATOR, EC_LEVEL, ECC_BLOCKS, ECC_PER_BLOCK, \
    ALPHANUMERIC, CHAR_COUNT_INDICATOR, NUM_DATA_CODEWORDS, REMAINDER_BITS

# pylint: disable=invalid-name


class MessageTooLong(Exception):
    """Raise when there is no version and ECC combination, that can hold a message."""


class RSEncoder(object):
    """Implementation of Reed Solomon error correction encoder.
    Computes error correction codewords, which can be done by 2 instance methods:
        get_ecc() - takes single data block.
        get_ecc_from_blocks() - takes sequence of nested data blocks.

    >>> RSEncoder(4, 2, 4).get_ecc([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    [3, 3, 12, 12]
    >>> RSEncoder(10).get_ecc(b'hello world')
    [237, 37, 84, 196, 253, 253, 137, 243, 168, 170]
    >>> RSEncoder(10).get_ecc([104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100])
    [237, 37, 84, 196, 253, 253, 137, 243, 168, 170]
    >>> RSEncoder(5).get_ecc_from_blocks([b'hello', [104, 101, 108, 108, 111]])
    [[99, 236, 136, 48, 85], [99, 236, 136, 48, 85]]

    There's also a static interleave_blocks() method providing the option
    to interleave codewords. It takes sequence of nested data blocks.

    >>> RSEncoder.interleave_blocks([[99, 236, 136, 48, 85], [99, 236, 136, 48, 85]])
    [99, 99, 236, 236, 136, 136, 48, 48, 85, 85]
    >>> RSEncoder.interleave_blocks([[1, 2, 3], [1, 2, 3, 4, 5]])
    [1, 1, 2, 2, 3, 3, 4, 5]
    """

    def __init__(self, ecc, prim_elem=2, power=8, prim_poly=None):
        """Input:   ecc - error correction codewords expected to being returned.
                    primitive element alpha and power for GF(alpha**power).
                    primitive polynomial - can be passed as an integer or left empty.
                    It will be computed automatically then."""
        # self.ecc = ecc
        self._gf_alpha = prim_elem
        self._gf_size = prim_elem ** power
        self._gf_px = prim_poly if prim_poly else self.find_prim_poly(prim_elem, power)
        self._logs, self._antilogs = self.log_antilog_tab(prim_elem, power, self._gf_px)

        self._generator = self._create_generator_poly(ecc)

    @staticmethod
    def find_prim_poly(alpha, power):
        """Brute force approach to find primitive polynomial p(x) for a given GF.
        It checks all polynomials from x to 2x where x is a field size GF(x).
        All GF elements are evaluated and when there's any duplicate value,
        then such a polynomial is rejected and another one is tested.

        Returns: First primitive polynomial found.

        ---- GF arithmetic background:
        Galois Field elements can be obtained by inductive multiplication of preceding
        element with primitive element alpha. It simply creates another power of alpha:
        Example for GF(2**4):
        α**0 = 0001; α**1 = 0010; α**2 = 0100; α**3 = 1000
        When we exceed max possible exponent of such a Galois Field we make use of gf
        arithmetic rules (inside GF, addition and subtractions are the same) and the fact
        that primitive element alpha is a root of p(x), (p(α) = 0) thus:
        Possible p(x) for GF(16) (which is to be found by this method) could be:
        p(x) = x**4 + x + 1, and p(α) = 0  so:
        α**4 + α + 1 = 0
        α**4 = α + 1 = 0011; α**5 = α**2 + α = 0110; α**6 = α**3 + α**2 = 1100

        Instead of this replacement we can also xor exceeding element with p(x):
        α**4 = 2**4 = 16 = 0b10000  <- exceeds 4-bit GF word.
        p(x) = x**4 + x + 1 = 19 = 0b10011  <- different forms of p(x)
        α**4 ^ p(x) = 16 ^ 19 = 3 = 0b0011  <- xor and result in GF again.
        ---- GF arithmetic background end:
        """
        poly_range = range(alpha ** power, alpha ** power * 2)
        for poly in poly_range:
            logs = {1}
            last_one = 1
            primitive = True
            for _ in range(1, alpha ** power - 1):
                result = last_one * alpha  # next element of the field.
                if result > (alpha ** power) - 1:
                    result ^= poly
                last_one = result
                if result in logs:
                    primitive = False
                    break
                else:
                    logs.add(result)

            if primitive:
                return poly
        return None

    @staticmethod
    def log_antilog_tab(alpha=2, power=8, prim_poly=285):
        """Generates all elements for Galois field GF(2**m).
        Default primitive polynomial for gf(256) is set to 285 (0b100011101),
        which represents p(x) = x**8 + x**4 + x**3 + x**2 + 1.

        Returns: logs as a list of elements where an index of the list represents current exponent.
                 antilogs as a dict where key is a result and value is an exponent
        """
        # set list with 1 at index 0. It's the first element and it represents α**0.
        # Multiply the preceding element by 2 to get the next one.
        # Every time the result is bigger than 255, xor with prim_poly.
        logs = [1]
        antilogs = {1: 0}
        for exp in range(1, alpha ** power - 1):
            result = logs[-1] * alpha  # next element of the field.
            if result > (alpha ** power) - 1:
                result ^= prim_poly
            logs.append(result)
            antilogs[result] = exp
        return logs, antilogs

    @staticmethod
    def interleave_blocks(blocks_list):
        """Input: 2D iterable. Output: list with data interleaved by 'columns'.
        Result is similar to numpy array.transpose().flatten(), however 'rows'
        may have different lengths here.
        """
        blocks_num = len(blocks_list)
        longest_block = max(len(b) for b in blocks_list)
        interleaved = []
        for idx in range(longest_block):
            for b in range(blocks_num):
                try:
                    interleaved.append(blocks_list[b][idx])
                except IndexError:
                    continue
        return interleaved

    def _gf_poly_multiply(self, poly1, poly2):
        """Multiply polynomials poly1 and poly2 according to GF multiplication arithmetic.
        Input polynomials' coefficients should represent alpha exponents.
        """
        # Indexes are exponents, so the result list has to be able to keep the sum
        # of the biggest exponents from both polynomials.
        result = [0] * (len(poly1) + len(poly2) - 1)
        for px_exp, p_coef in enumerate(poly1):
            for qx_exp, q_coef in enumerate(poly2):
                # Multiply x term from p with x term from q. It results in getting
                # x with these 2 terms exponents added. It's represented by an index
                # in the result list where coefficient is placed.
                result_x_exp = px_exp + qx_exp

                # Multiply coefficients representing alpha exponents (sum and divide modulo
                # to keep it inside a GF). Then get GF integer value taking advantage from
                # the precomputed logs table. Finally, combine like terms (same index)
                # by xoring them (addition in Galois Field).
                result[result_x_exp] ^= self._logs[(p_coef + q_coef) % (self._gf_size - 1)]

        # Back to alpha exponents representation. Coefficients are taken from
        # precomputed antilog table. There's no 0 representation.
        result = [self._antilogs[i] if i != 0 else 0 for i in result]
        return result

    def _create_generator_poly(self, ecc):
        """Input: integer as a number of parity symbols needed.
        Returns: List. Value for alpha exponent, index represents x power.

        Generator polynomial depends on a number of error correction codewords we want to add
        to a message and it will be a divisor of a message polynomial.
        It's a product of the following expression: (x-α**0)(x-α**1)(x-α**(ecc-1)).
        """
        generator = []
        for i in range(ecc):
            codeword = [i, 0]  # [α**ix**0, α**0x**1] index = x exponent; value = α exponent.
            if not generator:
                generator = codeword
                continue
            generator = self._gf_poly_multiply(generator, codeword)

        return generator

    def _gf_poly_divide(self, mes, gen):
        """Message polynomial is divided by generator polynomial.
        INPUT: Bytes or list of codewords in ascending order of exponents.
        OUTPUT: List of error correction codewords in ascending order of exponents.
                x**0 at index 0, x**5 at index 5.
        """
        # quotient = []  # uncomment this and 7th line below to get quotient as well.
        # Adding space for parity symbols in front of the message.
        mes = [0] * (len(gen) - 1) + list(mes)

        while len(mes) >= len(gen):
            lead_element = mes.pop()
            # quotient.append(lead_element)  # uncomment to get quotient as well.
            # Handles 0 issue, since there's no 0 key in antilogs table.
            if not lead_element:
                continue

            # Switch to alpha exponent representation for generator coeffs multiplication.
            # After it's done bring back integer representation for subtracting/adding (xoring).
            alpha_exp = self._antilogs[lead_element]
            divisor_gen = [self._logs[(coef + alpha_exp) % (self._gf_size - 1)] for coef in gen]

            # Xoring goes from the end of the message list (from the biggest x power) and
            # from the second biggest x power in generator list, since the last message element
            # is popped at the beginning of every loop step.
            for idx in range(2, len(gen) + 1):
                mes[-idx + 1] ^= divisor_gen[-idx]

        # Remainder of the message is returned.
        return mes

    def get_ecc(self, message):
        """Input: bytes or sequence of integers representing message.
        Output: list of integers representing error correction codewords.
        For upcoming calculations input sequence is reversed in order to let us treat
        indices as polynomial exponents. ECC output is reversed back.
        """
        message_reversed = message[::-1]
        ecc = self._gf_poly_divide(message_reversed, self._generator)
        return ecc[::-1]

    def get_ecc_from_blocks(self, blocks_list):
        """Input: sequence of nested data blocks.
        Output: list of nested error correction blocks
        """
        return list(self.get_ecc(block) for block in blocks_list)


class ModeEncoder(object):
    """Numeric, alphanumeric, latin-1, utf-8 and kanji encoding methods inside the class.
    auto_encode() static method will try to find the best encoding automatically.
    Here, "the best" means the least bytes needed to encode the message.
    """
    @staticmethod
    def numeric_encode(message):
        """Numbers are encoded in triples if possible. Bitstring length depends on
        size of such a group. For example: '222' has to be 10 bits long.
        '22' - 7 bits long and '2' - 4 bits long.
        """
        bin_repr = ''
        for i in range(0, len(message), 3):
            triple = message[i:i+3]
            triple_len = len(triple)
            try:
                bin_repr += format(int(triple), '0{}b'.format(triple_len * 3 + 1))
            except ValueError:
                return None
        return bin_repr

    @staticmethod
    def alphanum_encode(message):
        """If odd number of characters, then the last character is encoded with 6-bit binary string.
        Otherwise, two characters are paired and reduced to 11-bit numbers. It's done by multiplying
        first character code number with maximum code number increased by one. (Here it's 45).
        Then the second character code number is added to the result.
        """
        mes_len = len(message)
        total_alphanums = 45
        bin_repr = ''
        for idx in range(0, mes_len, 2):
            char0 = message[idx]
            try:
                if idx + 1 < mes_len:
                    char1 = message[idx + 1]
                    two_chars_pair = ALPHANUMERIC[char0] * total_alphanums + ALPHANUMERIC[char1]
                    bin_repr += format(two_chars_pair, '011b')
                else:
                    bin_repr += format(ALPHANUMERIC[char0], '06b')
            except KeyError:
                return None

        return bin_repr

    @staticmethod
    def latin1_encode(message):
        """Latin-1 encoding."""
        try:
            message = message.encode('latin-1')
        except UnicodeEncodeError:
            return None
        return ''.join(format(b, '08b') for b in list(message))

    @staticmethod
    def utf8_encode(message):
        """UTF-8 encoding. Handles kanji as well but with bigger byte cost,
        than kanji encoding itself."""
        message = message.encode('utf-8')
        return ''.join(format(b, '08b') for b in list(message))

    @staticmethod
    def kanji_encode(message):
        """Returns bitstring or None."""
        # TODO implement kanji encoder.
        pass

    @staticmethod
    def auto_encode(message):
        """Finds encoding with the least bytes needed to encode the message."""
        encoded = ModeEncoder.numeric_encode(message)
        if encoded:
            return ('num', 'Numeric'), encoded

        encoded = ModeEncoder.alphanum_encode(message)
        if encoded:
            return ('alphanum', 'Alphanumeric'), encoded

        encoded = ModeEncoder.latin1_encode(message)
        if encoded:
            return ('byte', 'Latin-1'), encoded

        # encoded = Encoder.kanji_encode(message)  # TODO uncomment block when kanji is done.
        # if encoded:
        #     return ('kanji', 'Kanji'), encoded

        return ('byte', 'UTF-8'), ModeEncoder.utf8_encode(message)


class Message(object):
    """The class encodes a message with the most appropriate encoding and adds all necessary
    binary prefixes and suffixes to follow qr code specification."""

    def __init__(self, mes, version=None, ec_level=2):
        assert ec_level in (0, 1, 2, 3), "Error correction level incorrect: {}. " \
                                         "Number between 0 and 3 expected.".format(ec_level)
        self.ec_lvl = ec_level
        self.mode, self.bitstring = ModeEncoder.auto_encode(mes)
        print('[+] {} encoding applied.'.format(self.mode[1]))

        # Number of characters in the input message or number of bytes
        # if a message is encoded as bytes.
        self.chars_total = len(self.bitstring) // 8 if self.mode[0] == 'byte' else len(mes)

        # QR code message always starts with a 4-bit mode indicator.
        self.mes = format(MODE_INDICATOR[self.mode[0]], '04b')

        self.ver = version
        self.max_capacity = None

        if self.ver:
            assert 0 < version < 41, "Version incorrect: {}. " \
                                     "Number between 1 and 40 expected.".format(version)
            self._validate_version_capacity(self.ver)
        else:
            self._auto_choose_version()

        print('[+] QR version: {} chosen.'.format(self.ver))
        print('[+] Error Correction Level: {} chosen.'.format(EC_LEVEL[self.ec_lvl]))

        self.__is_constructed = False

    def _auto_choose_version(self):
        """When version is not given, this method will try to find the most appropriate one.
        The priority is to keep initial error correction level. First all versions on a given
        ec level are checked and only when there's no success, a correction level is decreased.
        """
        for v in range(1, 41):
            try:
                return self._validate_version_capacity(v)
            except AssertionError:
                continue
        if self.ec_lvl > 0:
            self.ec_lvl -= 1
            return self._auto_choose_version()
        else:
            raise MessageTooLong("There's no QR code version able to hold this message.")

    def _validate_version_capacity(self, ver):
        """Checks if a version is capable to hold a message. Message bit length
        is compared with version max bit capacity. Message len is a sum of:
        - mode indicator length. It's constant and always equals 4.
        - Char count indicator len. It varies depending on mode and version.
        - encoded message length.
        """
        self.max_capacity = NUM_DATA_CODEWORDS[self.ec_lvl][ver] * 8
        mes_len = 4 + CHAR_COUNT_INDICATOR[self.mode[0]][ver] + len(self.bitstring)
        assert mes_len <= self.max_capacity, \
            'Message too long ({}) for a given version capacity ({})'.format(mes_len,
                                                                             self.max_capacity)
        self.ver = ver

    def _add_chr_count_indicator(self):
        """Character count indicator represents number of characters in the input message or
        number of bytes if a message is encoded as bytes. Its length depends on the encoding mode
        and qr version. All length values are mapped in the CHAR_COUNT_INDICATOR dictionary.
        """
        indi_len = CHAR_COUNT_INDICATOR[self.mode[0]][self.ver]
        self.mes += format(self.chars_total, '0{}b'.format(indi_len))

    def _add_terminator(self):
        """Add terminator of 4 zeros at most if message bit length is shorter than max capacity."""
        self.mes += '0' * min(4, self.max_capacity - len(self.mes))

    def _pad_zeros(self):
        """Pad zeros to make sure message bit length is a multiple of 8."""
        self.mes += '0' * (-len(self.mes) % 8)

    def _pad_bytes(self):
        """If a message is still too short, add pad bytes until the max capacity is reached.
        According to QR specification, these are the bytes: 11101100 00010001.
        """
        pad_bytes = ('11101100', '00010001')
        idx = 0
        while len(self.mes) < self.max_capacity:
            self.mes += pad_bytes[idx]
            idx ^= 1  # switch index between 0 and 1 to imitate pad_bytes chaining.

    def __construct(self):
        """Constructs base bitstring ready for ecc addition."""
        self._add_chr_count_indicator()
        self.mes += self.bitstring
        self._add_terminator()
        self._pad_zeros()
        self._pad_bytes()
        self.__is_constructed = True

    def _get_segments_sizes(self):
        """Returns sizes of blocks a message is to be split into. Values are taken
        from precomputed collection and depend on EC level and qr version.
        """
        blocks_number = ECC_BLOCKS[self.ec_lvl][self.ver]
        codewords = self.max_capacity // 8
        size1 = [codewords // blocks_number] * (blocks_number - codewords % blocks_number)
        size2 = [codewords // blocks_number + 1] * (codewords % blocks_number)
        return tuple(size1 + size2)

    def _make_segments(self):
        """Returns a message split into segments."""
        byte_mes = self.codewords_to_ints(self.mes)
        segments = []
        seg_start = 0
        for seg_size in self._get_segments_sizes():
            segments.append(byte_mes[seg_start:seg_start+seg_size])
            seg_start += seg_size

        return segments

    @staticmethod
    def codewords_to_ints(bitstr):
        """Converts bitstring into list of ints."""
        return list(int(bitstr[i:i + 8], 2) for i in range(0, len(bitstr), 8))

    @staticmethod
    def codewords_to_bitstr(int_list):
        """Converts list of ints into bitstring."""
        return ''.join(format(i, '08b') for i in int_list)

    def get_segments(self):
        """Returns message split into segments. Each segment is represented by a list of ints."""
        if not self.__is_constructed:
            self.__construct()

        return self._make_segments()

    def get_remainder_bits(self):
        """Returns remainder bits which have to be added to the final message.
        Number of bits is taken from precomputed collection and depends on qr version.
        """
        return '0' * REMAINDER_BITS[self.ver]

    def get_ecc_num(self):
        """Get precomputed number of error correction codewords for a given
        EC level and qr version.
        """
        return ECC_PER_BLOCK[self.ec_lvl][self.ver]

    def get_version(self):
        """Version getter."""
        return self.ver


if __name__ == "__main__":
    import doctest
    doctest.testmod()
