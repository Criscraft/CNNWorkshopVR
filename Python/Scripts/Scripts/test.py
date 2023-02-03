class myclass():
    def print_hello(self):
        print("hello")

a = myclass()
a.print_hello_old = a.print_hello
a.print_hello = lambda : print("bye")
a.print_hello()
a.print_hello_old()