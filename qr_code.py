#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""$ python qr_code.py -h
usage: qr_code.py [-h] [-v VERSION] [-e EC LEVEL] [-m MASK] [-s SCALE IMAGE]
                  [-i INPUT TEXT | -f INPUT FILE NAME] [-o OUTPUT IMAGE NAME]
                  [-z QUIET ZONE] [-d]

Create QR Code from text file [-f] or stdin [-i] and save as an image.
When version is to be chosen automatically, then it's worth noting,
that keeping declared error correction level will be a priority.
So, qr version will be increased first, before error correction level
is decreased in order to fit a message.

optional arguments:
  -h, --help            show this help message and exit
  -v VERSION            Valid values from 1 to 40. If omitted, version is computed automatically.
  -e EC LEVEL           Valid values from 0 to 3. Default 2.
                        0 - Level L (Low)         ~7% of codewords can be restored.
                        1 - Level M (Medium)      ~15% of codewords can be restored.
                        2 - Level Q (Quartile)    ~25% of codewords can be restored.
                        3 - Level H (High)        ~30% of codewords can be restored.
  -m MASK               Valid values from 0 to 7. Default None.
  -s SCALE IMAGE        Takes positive integer. Image is resized according to the value. Default 10.
  -i INPUT TEXT         Just write anything in "quotes" and the message will be processed.
  -f INPUT FILE NAME    Text file with a message to be encoded.
  -o OUTPUT IMAGE NAME  File name for QR code output image.
  -z QUIET ZONE         Takes positive integer. It's a width of white space around QR code.
  -d                    If the flag is present, QR code preview will be displayed, when done.
"""

from argparse import ArgumentParser, RawTextHelpFormatter, ArgumentTypeError

from message_constructor import Message, RSEncoder
from qr_grid import QR


# pylint: disable=invalid-name

def validate(val, num1=0, num2=float('inf')):
    """Validates CLI input arguments. Checks if a val is within a range(num1, num2)."""
    val = int(val)
    if not num1 <= val < num2:
        raise ArgumentTypeError("Value out of range: {}. "
                                "Should be between {} and {}.".format(val, num1, num2 - 1))
    return val


def get_args():
    """Process CLI input arguments."""
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter,
                            description="Create QR Code from text file [-f] or stdin [-i] "
                                        "and save as an image.\nWhen version is to be chosen "
                                        "automatically, then it's worth noting, \nthat keeping "
                                        "declared error correction level will be a priority.\n"
                                        "So, qr version will be increased first, before error "
                                        "correction level\nis lowered in order to fit a message.")

    parser.add_argument("-v", metavar="VERSION", type=lambda x: validate(x, 1, 41), default=None,
                        help="Valid values from 1 to 40. If omitted, "
                             "version is computed automatically.")
    parser.add_argument("-e", metavar="EC LEVEL", type=lambda x: validate(x, 0, 4), default=2,
                        help="Valid values from 0 to 3. Default 2.\n"
                             "0 - Level L (Low)         ~7%% of codewords can be restored.\n"
                             "1 - Level M (Medium)      ~15%% of codewords can be restored.\n"
                             "2 - Level Q (Quartile)    ~25%% of codewords can be restored.\n"
                             "3 - Level H (High)        ~30%% of codewords can be restored.\n")
    parser.add_argument("-m", metavar="MASK", type=lambda x: validate(x, 0, 8), default=None,
                        help="Valid values from 0 to 7. Default None.")
    parser.add_argument("-s", metavar="SCALE IMAGE", type=lambda x: validate(x, 1), default=10,
                        help="Takes positive integer. Image is resized according to this value. "
                             "Default 10.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-i", metavar="INPUT TEXT", type=str, default='',
                       help="Just write anything in \"quotes\" and the message will be processed.")
    group.add_argument("-f", metavar="INPUT FILE NAME", type=str,
                       help="Text file with a message to be encoded.")

    parser.add_argument("-o", metavar="OUTPUT IMAGE NAME", type=str, default=None,
                        help="File name for QR code output image.")
    parser.add_argument("-z", metavar="QUIET ZONE", type=lambda x: validate(x, 1), default=4,
                        help="Takes positive integer. It's a width of white space around QR code.")
    parser.add_argument("-d", action="store_true", default=False,
                        help="If the flag is present, QR code preview will be displayed, when done.")
    return parser.parse_args()


def create_qr_code(text_message, version=None, eclvl=2,
                   mask=None, fp=None, scale=10, qz=4, disp=True):
    """Main function, that wraps together all the steps needed to create and save QR code.
    INPUT:  text_message - simply a message.
            version - from 1 to 40. Optional, can be chosen automatically later.
            eclvl - from 0 to 3. Optional, default 2, stands for [Q] - 25% error correction.
            mask - from 0 to 7. Optional, the best one can be chosen automatically later.
            fp - File name for QR code output image. If None, QR image won't be saved.
            scale - Optional. Default 10. Image is resized according to this value.
            qz - Quiet Zone. Default 4. It's a width of white space around QR code.
            disp - Bool. Default True. QR code preview will be displayed or not, when done.
    """
    m = Message(text_message, version=version, ec_level=eclvl)
    ecc_num = m.get_ecc_num()
    version = m.get_version()
    rs = RSEncoder(ecc_num, prim_poly=285)

    mes_blocks = m.get_segments()
    ecc_blocks = rs.get_ecc_from_blocks(mes_blocks)

    full_mes = rs.interleave_blocks(mes_blocks) + rs.interleave_blocks(ecc_blocks)
    full_mes = m.codewords_to_bitstr(full_mes)
    full_mes += m.get_remainder_bits()

    qr = QR(version)
    qr.insert_message(full_mes)
    qr.apply_mask(eclvl, mask)
    qr.add_quiet_zone(qz)
    qr.save_qr(fp, scale, disp)


def get_text_from_file(filepath):
    """Returns a message text for QR taken from a text file."""
    with open(filepath, 'r') as f:
        return f.read()


if __name__ == "__main__":
    args = get_args()
    if args.f:
        txt_mes = get_text_from_file(args.f)
    else:
        txt_mes = args.i

    create_qr_code(txt_mes, args.v, args.e, args.m, args.o, args.s, args.z, args.d)
