class A:
    def __init__(self, *args, kwarg3=1, kwarg4=2, **kwargs):
        print(args)
        print(kwargs)
class B(A):
    def __init__(self, *args, kwarg1=1, kwarg2=2, **kwargs):
        super().__init__(*args, **kwargs)
        print(args)
        print(kwargs)

if __name__ == "__main__":
    B(1,2, kwarg1=1, kwarg2=2, kwarg3=3, kwarg4=4)
