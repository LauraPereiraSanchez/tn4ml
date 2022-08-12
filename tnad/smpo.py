import itertools
import quimb as qu
import quimb.tensor as qtn
from typing import Tuple
import autoray as a
import math
from quimb.tensor.tensor_1d import TensorNetwork1DOperator, TensorNetwork1DFlat, TensorNetwork1D


class SpacedMatrixProductOperator(TensorNetwork1DOperator, TensorNetwork1DFlat, TensorNetwork1D, qtn.TensorNetwork):
    """A MatrixProductOperator with a decimated number of output indices.

    Parameters
    ----------
    spacing : int
        Spacing paramater, or space between output indices in number of sites.
    embed_dim : int
    physical_dim : int
    """

    _EXTRA_PROPS = ("_site_tag_id", "_upper_ind_id", "_lower_ind_id", "_L", "_spacing")

    def __init__(self, arrays, shape="lrud", site_tag_id="I{}", tags=None, upper_ind_id="k{}", lower_ind_id="b{}", bond_name="", **tn_opts):
        if isinstance(arrays, SpacedMatrixProductOperator):
            super().__init__(arrays)
            return

        arrays = tuple(arrays)
        self._L = len(arrays)
        # process site indices
        self._upper_ind_id = upper_ind_id
        self._lower_ind_id = lower_ind_id
        upper_inds = map(upper_ind_id.format, range(self.L))
        lower_inds = map(lower_ind_id.format, range(self.L))
        # process tags
        self._site_tag_id = site_tag_id
        site_tags = map(site_tag_id.format, range(self.L))
        if tags is not None:
            tags = (tags,) if isinstance(tags, str) else tuple(tags)
            site_tags = tuple((st,) + tags for st in site_tags)

        self.cyclic = qtn.array_ops.ndim(arrays[0]) == 4
        dims = [x.ndim for x in arrays]
        #hasoutput = [x.shape[-1]>1 for x in arrays]
        #q = hasoutput.count(True)
        q = dims.count(4) + 1 + (1 if dims[-1]==3 else 0)
        print(q)
        #print(hasoutput)
        self._spacing = math.ceil((self.L - 1) / (q-1)) - 1
        print(self.spacing)
        #self._spacing = ([x.ndim for x in arrays].count(4) + self.L - 1) // self.L
        
        # process orders
        lu_ord = tuple(shape.replace("r", "").replace("d", "").find(x) for x in "lu")
        lud_ord = tuple(shape.replace("r", "").find(x) for x in "lud")
        rud_ord = tuple(shape.replace("l", "").find(x) for x in "rud")
        lru_ord = tuple(shape.replace("u", "").find(x) for x in "lru")
        lrud_ord = tuple(map(shape.find, "lrud"))

        if self.cyclic:
            if (self.L - 1) % self.spacing == 0:
                last_ord = lrud_ord
            else:
                last_ord = lru_ord
        else:
            if (self.L - 1) % self.spacing == 0:
                last_ord = lud_ord
            else:
                last_ord = lu_ord

        #orders = [rud_ord if not self.cyclic else lrud_ord, *[lrud_ord for i in range(1, self.L - 1)], last_ord]
        orders = [rud_ord if not self.cyclic else lrud_ord, *[lrud_ord if i % self.spacing == 0 else lru_ord for i in range(1, self.L - 1)], last_ord]


        # process inds
        cyc_bond = (qtn.rand_uuid(base=bond_name),) if self.cyclic else ()
        nbond = qtn.rand_uuid(base=bond_name)
        
        inds = []
        inds += [(*cyc_bond, nbond, next(upper_inds), next(lower_inds))]
        pbond = nbond

        for i in range(1, self.L - 1):
            nbond = qtn.rand_uuid(base=bond_name)

            if i % self.spacing == 0:
                curr_down_id = [lower_ind_id.format(i)]
            else:
                curr_down_id = []

            inds += [(pbond, nbond, next(upper_inds), *curr_down_id)]
            pbond = nbond

        last_down_ind = [lower_ind_id.format(self.L - 1)] if (self.L - 1) % self.spacing == 0 else []
        inds += [(pbond, *cyc_bond, next(upper_inds), *last_down_ind)]

        # for array, site_tag, ind, order in zip(arrays, site_tags, inds, orders):
        #     print(f'Array shape: {array.shape}')
        #     print(f'Order: {order}')
        #     print(f'Site tag: {site_tag}')
        #     print(f'Ind: {ind}')
            
        tensors = [qtn.Tensor(data=a.transpose(array, order), inds=ind, tags=site_tag) for array, site_tag, ind, order in zip(arrays, site_tags, inds, orders)]

        super().__init__(tensors, virtual=True, **tn_opts)

    def rand(n: int, spacing: int, bond_dim: int = 4, phys_dim: Tuple[int, int] = (2, 2), cyclic: bool = False, init_func: str = "normal", **kwargs):
        arrays = []
        for i, hasoutput in zip(range(n), itertools.cycle([True, *[False] * (spacing - 1)])):
            if hasoutput:
                shape = (bond_dim, bond_dim, *phys_dim)
                if not cyclic:
                    if i==0: shape = (bond_dim, *phys_dim)
                    if i==n-1: shape = (bond_dim, *phys_dim)
                arrays.append(qu.gen.rand.randn(shape, dist=init_func))
            else:
                shape = (bond_dim, bond_dim, phys_dim[0])
                if i==n-1 and not cyclic: shape = (bond_dim, phys_dim[0])
                arrays.append(qu.gen.rand.randn(shape, dist=init_func))
        mpo = SpacedMatrixProductOperator(arrays, **kwargs)
        mpo.draw()
        mpo.compress(form='flat', max_bond=bond_dim) # limit bond_dim
        return mpo

    @property
    def spacing(self) -> int:
        return self._spacing

    def apply():
        pass

    def trace():
        pass
