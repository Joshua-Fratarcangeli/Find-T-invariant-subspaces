import copy

import pandas as pd
import numpy as np
from numpy.polynomial import Polynomial as P

class Matrix:
	"""
	-- Public Attributes --
	n: int - The number of rows/columns this matrix has.
	det: int - The determinant of the matrix.
	rows: int - The number of rows this natrix has.
	-- Public Methods --
	getData() -> np.ndarray - Returns an array representing the data stored in this matrix.
	polynomial() -> Polynomial - Returns a Polynomial object represebting the characteristic polynomial of the matrix representing a linear trabsformation. If the characteristic polynomial does not split, return the Null Polynomial.
	setData(data) -> None - Sets the Matrix's data to the provided numpy array or list-like in 2D. Data is converted/stores as a ndarray.
	charPoly() -> Polynomial
	Returns the characteristic polynomial of the Matrix, as the deterninant of M - xI, where I is the nxn Identity Matrix.
	-- Private Attributes --
	_data: ndarray - A NunpPy n-dimenaional array which holds the matrix's underlying data. The first level of the array represents the rows, whilst the second level represents the columns.
	-- Private Methods --
	_det() -> int - Calculates the determinant of the matrix and updates the det attribute with the value, and returns it. The method is called after modifying the matrix to maintain invariance in between the det attribute and the actual determinant of the matrix.
	-- Something invariants --
	- The det attribute of the Matrix is equal to the actual determinant of the Matrix's data array.
	- The number of rows/columns of the _data attribute must be equal to n.
	
	This class is a representation of a matrix of integers as a matricized form of a linear transformation T: V -> V of any fixed basis for V, where V is finite dimensional. The attributes and methods of the matrix allows data to be pulled and set in the form of NumPy arrays, as well as retieving mathematic properties such as the determinant and charcteristic polynomials as integers and Polynomial objects, respectively.
	"""
	
	def __init__(self, n, data=None):
		"""
		:n: int - The number of rows and columns of the Matrix
		:data: ndarray or list-like of depth 2 - Data to initialize the Matrix with. Converts to a 2-dimensional array.
		"""
		self.n = n
		if data is not None:
			data = np.array(data)
			self._data = data
		else:
			data = [[0]*n]*n
			self._data = np.array(data)
			
		self._det()
		
	def __str__(self):
		str = ""
		for row in np.nditer(self._data, flags=["external_loop"], order='F', itershape=(self.n, self.n)):
			str += row.__str__() + "\n"
		return str
		
	def __add__(self, other):
		if isinstance(other, Matrix):
			data = self._data + other._data
		else:
			data = self._data + other
		return Matrix(self.n, data)
		
	def __sub__(self, other):
		if isinstance(other, Matrix):
			data = self._data - other._data
		else:
			data = self._data - other
		return Matrix(self.n, data)
		
	def __mul__(self, other):
		if isinstance(other, Matrix):
			data = np.dot(self._data, other._data)
		else:
			data = np.dot(self._data, other)
		return Matrix(self.n, data)
		
	def getData(self):
		return self._data.copy.copy()
		
	def setData(self, data):
		# Todo check and add new invariant to confirm entries of data lists are of the proper number of rows/columns.
		if not isinstance(data, np.array):
			data = np.array(data, int32)
		self._data = data
		self._det()
		
	def charPoly(self):
		"""
		Returns the characteristic Polynomial of this matrix M. Calculated by det(M - xI), where I is the nxn Identity Matrix.
		"""
		ct = self - Matrix.identity(self.n) * P([0, 1])
		return ct.det
	
	def _det(self):
		"""
		Finds the determinant of the Matrix, setting the det attribute to this value and returning it. Done through cofactor expansion along the first row.
		"""
		det = self._detHelper(self.n, self._data)
		self.det = det
		return det
		
	def _detHelper(self, n, data):
		"""
		Params
		:n: int - Size of the array.
		:data: Array-like - Data to find the determinant od
		Calculates and returns the determinant of a square nxn matrix over R. Matrix is represented by a 2-D array-like structure of size n.
		"""
		if n == 0:
			return 0
		if n == 1:
			return data[0][0]
		if n == 2:
			return (data[0,0] * data[1,1]) - (data[0,1] * data[1,0])

		det = 0
		for i in range(n):
			droppedI = list(range(n))
			droppedI.remove(i)
			det += (-1)**i * data[0][i]  * self._detHelper(n-1, data[np.ix_(range(1, n), droppedI)])
		return det
	
	@classmethod	
	def identity(self, n):
		"""
		:n: int - size of the Identity Matrix to return
		
		Returns an nxn Identity Matrix object, containing 1's in the diagonal entries and 0s in all others.
		"""
		data = []
		for i in range(n):
			row = []
			for j in range(n):
				if i == j:
					row.append(1)
				else:
					row.append(0)
			data.append(row)
		
		return Matrix(n, data)
		
def linearFactor(p):
	"""
	:p : Polynomial - Polynomial to factor
	
	Factors the given Polynomial into its linear roots, and returns a dict of its factors, with factors as keys and multiplicities of each factor as values. Returns the remainder under the key "r_".
	"""
	d = {}
	roots = p.roots()
	pDiv = p
	for root in roots:
		root = complex(round(root.real, 4), round(root.imag, 4))
		pRoot = P([root, -1])
		rem = pDiv % pRoot
		pDiv = pDiv // pRoot
		if root not in d:
			d[root] = 1
		else:
			d[root] += 1
	d["r_"] = rem
	return d
	
def getDimESpace(matrix, eval):
	"""
	:matrix : Matrix representation of linear transformation
	:eval : known eigenvalue of thr matrix
	
	Finds and returns the dimension of the Eigenspace corresponding to the eigenvalue eval in the matrx argument.
	Precondition: eval is an eigenvalue of the matrix.
	"""
	nullMatrix = matrix - Matrix.identity(matrix.n) * eval.real # irrelevant to computation on whether thisbis handled as a complex number, sincebour matrices are real
	mat = nullMatrix + Matrix(3,[[1,0,0],[0,0,0],[0,0,0]])
	print(mat)
	val = np.linalg.matrix_rank(nullMatrix._data, tol=None)
	# Sorry, I wanted to implement Gaussian elimination here but I don't have enough time this week
	
	return matrix.n - val 

def getNumInvSubs(matrix):
	"""
	:mat : Matrix - Matrix representation of a linear transformation on Complex Vector Space V -> V.
	
	Takes the Matrix representation of a linear transformation T on an n-dimensional Vector Space V to V and returns an integer representing the number of T-invariant subspaces that exist on that Vector Space. If the characteristic polynomial of T splits, the result is the sum of combinations of all distinct eigenvalues (n k) for all 0 <= k <= n, which is 2 ** n, 2 if the characteristic polynomial does not split (which should never happen in a complex vector space), or 1 if V is the 0 vector space.
	"""
	if matrix.n == 0:
		return 1
	cp = matrix.charPoly()
	factorDict = linearFactor(cp)
	print(factorDict)
	if factorDict["r_"] == P([complex(0,0)]):
		numESpaces = 0
		keys = list(factorDict.keys())
		keys.remove("r_")
		for eval in keys:
			numESpaces += getDimESpace(matrix, eval)
		return 2 ** numESpaces
	return 2

if __name__ == "__main__":
		inp = None
		while inp != "q":
			print("Enter a  comma-separated real square matrix by row: ('q'' to quit, empty line to submit matrix)")
			rows = []
			while inp != "" or (len(rows) > 0 and len(rows) != len(rows[0])):
				inp = input()
				if inp == "":
					continue
				try:
					row = inp.split(",")
					print(row)
					if len(rows) > 0 and len(row) != len(rows[0]):
						print("Matrices must be square.")
						continue
					for i in range(len(row)):
						row[i] = float(row[i])
					rows.append(row)
				except:
					print("Invalid entries, try again.")
					continue
			print(rows)
			numInvSubspaces = getNumInvSubs(Matrix(len(rows), rows))
			print("Your Matrix (or Linear trabsformation) has a total of %s T-invariant Subspaces in a %s-dimensional real-isomorphism to a finite vector space of the same dimension." % (numInvSubspaces, len(rows)))
			inp = None
		
	