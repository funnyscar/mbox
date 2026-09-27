import jax
import jax.numpy as jnp

class Tensor:
	def __init__(self, data, 
		_children=(), 
		_op="", 
		label="", 
		dtype = jnp.float32, 
		requires_grad=True
	):
		self._children = set(_children)
		self._op = _op
		self.label = label

		# pytorch based
		self.data, self.shape = self._clean_data(data) # new lightweight Tensor object
		self.dtype = dtype # (float32, int64, etc.)
		self.device = self.data.device if self.shape else "cpu"
		self.requires_grad = requires_grad # default assumes non-constant
		self.grad = jnp.zeros(self.shape) if self.shape else 0
		self.grad_fn = lambda: None
		self.is_leaf = (len(self._children) == 0)
		self.T = self.data.T if self.shape else self.data
		self.ndim = len(self.shape)
		self.is_cuda = "Cuda" in str(self.data.device) if self.shape else False

	# pytorch
	def size(self):
		return self.shape
	
	def numel(self):
		return jnp.prod(self.data).item()

	@staticmethod
	def devices():
		return jax.devices()
		
	def to(self, device):
		jax.device_put(self.data, device)


	##############################################################
	
	def __repr__(self):
		return f"Tensor({self.data}, shape={self.shape})"

	def _clean_data(self, other):
		# inputs: int, float, np.array, [], jnp.array, Tensor
		# output: scalar or jnp.ndarray
		if isinstance(other, (int, float)):
			return other, ()
		if isinstance(other, np.ndarray):
			return jnp.array(other), other.shape
		if isinstance(other, list):
			return jnp.array(other), (len(other),)
		if isinstance(other, Tensor):
			return other.data, other.shape
		if isinstance(other, jnp.ndarray):
			return other, other.shape
		raise ValueError(f"Unknown value for Tensor.data: {other}")

	
	def _clean_tensor(self, other):
		# inputs: int, float, np.array, [], jnp.array, Tensor
		# output: Tensor
		if isinstance(other, (int, float)):
			return Tensor(jnp.full(self.shape, other), dtype=type(other))
		if isinstance(other, (np.ndarray, list, jnp.ndarray)): 
			return Tensor(other)
		if isinstance(other, Tensor):
			return other
		raise ValueError(f"Unknown value: {other}")


	##############################################################


	def __add__(self, other):
		v = self._clean_tensor(other)
		out = Tensor(self.data + v.data, (self, v), _op="+")

		def _backwards():
			if not self.requires_grad: return
			self.grad += out.grad * 1.0
			v.grad += out.grad * 1.0
		
		out.grad_fn = _backwards
		return out
	
	def __mul__(self, other):
		v = self._clean_tensor(other)
		out = Tensor(self.data * v.data, (self, v), _op="*")

		def _backwards():
			if not self.requires_grad: return
			self.grad += out.grad * v.data
			v.grad += out.grad * self.data
		
		out.grad_fn = _backwards
		return out
		
	def __pow__(self, other):
		x = self.data
		v = self._clean_tensor(other)
		k = v.data
		out = Tensor(x**k, (self,v), _op="**")
		def _backwards():
			if not self.requires_grad: return
			self.grad += out.grad * k * (x**(k-1))		# self**other
			v.grad += out.grad * out.data * jnp.log(x)
		out.grad_fn = _backwards
		return out
	def __rmul__(self, other):
		return self * other
	def __truediv__(self, other):
		return self * other**-1
	def __radd__(self, other): # other + self
		return self + other
	def __neg__(self):
		return self * -1
	def __sub__(self, other):
		return self + (-other)
	def __rsub__(self, other):
		return other + (-self)
	def __iter__(self):
		return iter(self.data)
	def __getitem__(self, idx):
		out = Tensor(self.data[idx], _children=(self,))
		def _backward():
			self.grad = self.grad + jnp.zeros_like(self.data).at[idx].add(out.grad)
		out.grad_fn = _backward
		return out


	def backward(self):
		topo = []
		visited = set()
		def build_topo(v):
			if v not in visited:
				visited.add(v)
				for child in v._children:
					build_topo(child)
				topo.append(v)
		build_topo(self)

		self.grad = jnp.ones(self.shape)
		for node in reversed(topo):
			node.grad_fn()
	

	##############################################################

	# DON'T fully understand
	def sum(self, axis=None, keepdims=False):
		out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), _children=(self,))
		def _backward():
			g = out.grad
			if axis is not None and not keepdims:
				g = jnp.expand_dims(g, axis)          # restore the summed axis as size 1
			self.grad = self.grad + jnp.broadcast_to(g, self.data.shape)
		out.grad_fn = _backward
		return out

	def max(self, axis=None, keepdims=False):
		out = Tensor(self.data.max(axis=axis, keepdims=keepdims), _children=(self,))
		def _backward():
			m, g = out.data, out.grad
			if axis is not None and not keepdims:
				m = jnp.expand_dims(m, axis)
				g = jnp.expand_dims(g, axis)
			mask = (self.data == m)                                   # where the max was
			mask = mask / mask.sum(axis=axis, keepdims=True)          # split ties evenly
			self.grad = self.grad + mask * g                          # broadcasts g
		out.grad_fn = _backward
		return out

	def mean(self, axis=None, keepdims=False):
		n = self.data.size if axis is None else self.data.shape[axis]
		return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

	def exp(self):
		x = self.data
		v = jnp.exp(x)
		out = Tensor(v, (self,), _op="exp")
		def _backwards():
			if not self.requires_grad: return
			self.grad += out.grad * v
		out.grad_fn = _backwards
		return out
	
	def tanh(self):
		x = self.data
		t = jnp.tanh(x)
		out = Tensor(t, (self,), _op="tanh")
		def _backwards():
			if not self.requires_grad: return
			self.grad += out.grad * (1 - t**2)
		out.grad_fn = _backwards

		return out
	
	def relu(self):
		x = self.data
		v = jnp.maximum(x, jnp.zeros(x.shape)) 
		out = Tensor(v, (self,), _op="relu")
		def _backwards():
			if not self.requires_grad: return
			self.grad = out.grad * (x > 0)
		out.grad_fn = _backwards
		return out

	def sigmoid(self):
		x = self.data
		v = 1 / (1 + jnp.exp(-x))
		out = Tensor(v, (self,), _op="sigmoid")
		def _backwards():
			if not self.requires_grad: return
			self.grad = out.grad * v * (1-v)
		out.grad_fn = _backwards
		return out