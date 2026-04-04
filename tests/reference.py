import numpy

a = numpy.arange(5)
b = numpy.arange(5)

A = numpy.random.random((5, 5))

print(numpy.einsum('ij->', A))
print(A.sum())