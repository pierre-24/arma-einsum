import numpy

a = numpy.arange(5)
b = numpy.arange(5)

print(numpy.einsum('i,j->ij', a, b))