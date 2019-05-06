import qrcode


class Generator:

    def __init__(self):
        pass

    @staticmethod
    def generate(path='https://github.com/Intel-out-side'):
        img = qrcode.make(path)

        print(type(img))
        print(img.size)

        img.save("./qrcode_test.png")


if __name__ == '__main__':
    Generator.generate()
