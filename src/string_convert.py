import sys


def str_convert(string: str):
    # translation_table = str.maketrans(
    #     "!\"#$%&\'()*+,-./:;<=>?\\]^_`{|}~ ",
    #     "_______________________________"
    # )
    translation_table = {33: 95, 34: 95, 35: 95, 36: 95, 37: 95, 38: 95, 39: 95, 40: 95,
                         41: 95, 42: 95, 43: 95, 44: 95, 45: 95, 46: 95, 47: 95, 58: 95,
                         59: 95, 60: 95, 61: 95, 62: 95, 63: 95, 92: 95, 93: 95, 94: 95,
                         95: 95, 96: 95, 123: 95, 124: 95, 125: 95, 126: 95, 32: 95}
    output = ascii(string.casefold()).translate(translation_table)
    if not output[1].isdecimal():
        output = output.removeprefix("_")
    output = output.removesuffix("_")
    return output


if __name__ == '__main__':
    names = ['Case_control', '0His', 'Ile', 'Leu', 'Lys', 'Met', 'Phe', 'Ser', 'Thr',
             'Trp', 'Val', 'PC ae C34:3', 'lysoPC a C18:2', 'SM C18:0', '魔入りました！入間くん']
    for i in names:
        print(str_convert(i))
    sys.exit()
