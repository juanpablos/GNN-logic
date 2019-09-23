from .label_generator import Tagger


class None_down:
    def __ge__(self, value):
        return False

    def __gt__(self, value):
        return False

    def __le__(self, value):
        return True

    def __lt__(self, value):
        return True


class None_up:
    def __ge__(self, value):
        return True

    def __gt__(self, value):
        return True

    def __le__(self, value):
        return False

    def __lt__(self, value):
        return False
