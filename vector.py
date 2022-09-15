#Class Taken from http://www.pygame.org/wiki/2DVectorClass and modified
#Author unknown
########################################################################
import operator
import math

class Vector(object):
    """2d vector class, supports vector and scalar operators,
       and also provides a bunch of high level functions
       """
    #Constants
    rad90deg = math.pi/2
    rad180deg = math.pi
    rad270deg = math.pi*3/2
    rad360deg = math.pi*2

    __slots__ = ['x', 'y']

    def __init__(self, x_or_pair, y = None):
        if y == None:
            self.x = x_or_pair[0]
            self.y = x_or_pair[1]
        else:
            self.x = x_or_pair
            self.y = y

    def __len__(self):
        return 2

    def __getitem__(self, key):
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        else:
            raise IndexError("Invalid subscript "+str(key)+" to Vector")

    def __setitem__(self, key, value):
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value
        else:
            raise IndexError("Invalid subscript "+str(key)+" to Vector")

    # String representaion (for debugging)
    def __repr__(self):
        return 'Vector(%s, %s)' % (self.x, self.y)

    # Comparison
    def __eq__(self, other):
        if hasattr(other, "__getitem__") and len(other) == 2:
            return self.x == other[0] and self.y == other[1]
        else:
            return False

    def __ne__(self, other):
        if hasattr(other, "__getitem__") and len(other) == 2:
            return self.x != other[0] or self.y != other[1]
        else:
            return True

    def __nonzero__(self):
        return bool(self.x or self.y)

    def __bool__(self):
        return self.__nonzero__()

    # Generic operator handlers
    def _o2(self, other, f):
        "Any two-operator operation where the left operand is a Vector"
        if isinstance(other, Vector):
            return Vector(f(self.x, other.x),
                         f(self.y, other.y))
        elif (hasattr(other, "__getitem__")):
            return Vector(f(self.x, other[0]),
                         f(self.y, other[1]))
        else:
            return Vector(f(self.x, other),
                         f(self.y, other))

    def _r_o2(self, other, f):
        "Any two-operator operation where the right operand is a Vector"
        if (hasattr(other, "__getitem__")):
            return Vector(f(other[0], self.x),
                         f(other[1], self.y))
        else:
            return Vector(f(other, self.x),
                         f(other, self.y))

    def _io(self, other, f):
        "inplace operator"
        if (hasattr(other, "__getitem__")):
            self.x = f(self.x, other[0])
            self.y = f(self.y, other[1])
        else:
            self.x = f(self.x, other)
            self.y = f(self.y, other)
        return self

    # Addition
    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        elif hasattr(other, "__getitem__"):
            return Vector(self.x + other[0], self.y + other[1])
        else:
            return Vector(self.x + other, self.y + other)
    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, Vector):
            self.x += other.x
            self.y += other.y
        elif hasattr(other, "__getitem__"):
            self.x += other[0]
            self.y += other[1]
        else:
            self.x += other
            self.y += other
        return self

    # Subtraction
    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y)
        elif (hasattr(other, "__getitem__")):
            return Vector(self.x - other[0], self.y - other[1])
        else:
            return Vector(self.x - other, self.y - other)
    def __rsub__(self, other):
        if isinstance(other, Vector):
            return Vector(other.x - self.x, other.y - self.y)
        if (hasattr(other, "__getitem__")):
            return Vector(other[0] - self.x, other[1] - self.y)
        else:
            return Vector(other - self.x, other - self.y)
    def __isub__(self, other):
        if isinstance(other, Vector):
            self.x -= other.x
            self.y -= other.y
        elif (hasattr(other, "__getitem__")):
            self.x -= other[0]
            self.y -= other[1]
        else:
            self.x -= other
            self.y -= other
        return self

    # Multiplication
    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x*other.x, self.y*other.y)
        if (hasattr(other, "__getitem__")):
            return Vector(self.x*other[0], self.y*other[1])
        else:
            return Vector(self.x*other, self.y*other)
    __rmul__ = __mul__

    def __imul__(self, other):
        if isinstance(other, Vector):
            self.x *= other.x
            self.y *= other.y
        elif (hasattr(other, "__getitem__")):
            self.x *= other[0]
            self.y *= other[1]
        else:
            self.x *= other
            self.y *= other
        return self

    # Division
    def __div__(self, other):
        return self._o2(other, operator.div)
    def __rdiv__(self, other):
        return self._r_o2(other, operator.div)
    def __idiv__(self, other):
        return self._io(other, operator.div)

    def __floordiv__(self, other):
        return self._o2(other, operator.floordiv)
    def __rfloordiv__(self, other):
        return self._r_o2(other, operator.floordiv)
    def __ifloordiv__(self, other):
        return self._io(other, operator.floordiv)

    def __truediv__(self, other):
        return self._o2(other, operator.truediv)
    def __rtruediv__(self, other):
        return self._r_o2(other, operator.truediv)
    def __itruediv__(self, other):
        return self._io(other, operator.floordiv)

    # Modulo
    def __mod__(self, other):
        return self._o2(other, operator.mod)
    def __rmod__(self, other):
        return self._r_o2(other, operator.mod)

    def __divmod__(self, other):
        return self._o2(other, operator.divmod)
    def __rdivmod__(self, other):
        return self._r_o2(other, operator.divmod)

    # Exponentation
    def __pow__(self, other):
        return self._o2(other, operator.pow)
    def __rpow__(self, other):
        return self._r_o2(other, operator.pow)

    # Bitwise operators
    def __lshift__(self, other):
        return self._o2(other, operator.lshift)
    def __rlshift__(self, other):
        return self._r_o2(other, operator.lshift)

    def __rshift__(self, other):
        return self._o2(other, operator.rshift)
    def __rrshift__(self, other):
        return self._r_o2(other, operator.rshift)

    def __and__(self, other):
        return self._o2(other, operator.and_)
    __rand__ = __and__

    def __or__(self, other):
        return self._o2(other, operator.or_)
    __ror__ = __or__

    def __xor__(self, other):
        return self._o2(other, operator.xor)
    __rxor__ = __xor__

    # Unary operations
    def __neg__(self):
        return Vector(operator.neg(self.x), operator.neg(self.y))

    def __pos__(self):
        return Vector(operator.pos(self.x), operator.pos(self.y))

    def __abs__(self):
        return Vector(abs(self.x), abs(self.y))

    def __invert__(self):
        return Vector(-self.x, -self.y)

    # vectory functions
    def get_length_sqrd(self):
        return self.x**2 + self.y**2

    def get_length(self):
        return math.sqrt(self.x**2 + self.y**2)

    def faced_right(self, angle = None):
        if angle == None:
            if isinstance(self, Vector):
                angle = self.get_positive_angle()
            else:
                angle = self
        return (angle < Vector.rad90deg or angle > Vector.rad270deg)

    def faced_up(self, angle = None):
        if angle == None:
            if isinstance(self, Vector):
                angle = self.get_positive_angle()
            else:
                angle = self
        return (angle <= Vector.rad180deg)

    def __setlength(self, value):
        length = self.get_length()
        self.x *= value/length
        self.y *= value/length
    length = property(get_length, __setlength, None, "gets or sets the magnitude of the vector")

    def rotate(self, radians):
        cos = math.cos(radians)
        sin = math.sin(radians)
        x = self.x*cos - self.y*sin
        y = self.x*sin + self.y*cos
        self.x = x
        self.y = y
        return self

    def rotated(self, radians):
        cos = math.cos(radians)
        sin = math.sin(radians)
        x = self.x*cos - self.y*sin
        y = self.x*sin + self.y*cos
        return Vector(x, y)

    def rotate_degree(self, angle_degrees):
        return self.rotate(math.radians(angle_degrees))

    def rotated_degree(self, angle_degrees):
        return self.rotated(math.radians(angle_degrees))


    def get_angle_degree(self):
        return math.degrees(self.get_angle())

    def get_angle(self):
        if (self.get_length_sqrd() == 0):
            return 0
        return math.atan2(self.y, self.x)

    def get_positive_angle(self, angle = None):
        if angle == None:
            if isinstance(self, Vector):
                angle = self.get_positive_angle()
            else:
                angle = self

        angle = math.copysign((abs(angle) % Vector.rad360deg), angle)
        if (angle < 0):
            angle += Vector.rad360deg

        return angle

    def __setangle(self, angle_degrees):
        self.x = self.length
        self.y = 0
        self.rotate(angle_degrees)
    angle = property(get_angle, __setangle, None, "gets or sets the angle of a vector")

    def get_angle_between(self, other):
        cross = self.x*other[1] - self.y*other[0]
        dot = self.x*other[0] + self.y*other[1]
        return math.degrees(math.atan2(cross, dot))

    def normalized(self):
        length = self.length
        if length != 0:
            return self/length
        return Vector(self)

    def normalize_return_length(self):
        length = self.length
        if length != 0:
            self.x /= length
            self.y /= length
        return length

    def round(self, len = 0):
        self.x = round(self.x, len)
        self.y = round(self.y, len)
        return self

    def rounded(self, len = 0):
        result = Vector(self)
        result.round(len)
        return result

    def toInt(self):
        self.x = int(self.x)
        self.y = int(self.y)
        return self

    def toInted(self):
        result = Vector(self)
        result.toInt()
        return result

    def toTuple(self):
        return (self.x,self.y)

    def perpendicular(self):
        return Vector(-self.y, self.x)

    def perpendicular_normal(self):
        length = self.length
        if length != 0:
            return Vector(-self.y/length, self.x/length)
        return Vector(self)

    def dot(self, other):
        return float(self.x*other[0] + self.y*other[1])

    def get_distance(self, other):
        return math.sqrt((self.x - other[0])**2 + (self.y - other[1])**2)

    def get_dist_sqrd(self, other):
        return (self.x - other[0])**2 + (self.y - other[1])**2

    def projection(self, other):
        other_length_sqrd = other[0]*other[0] + other[1]*other[1]
        projected_length_times_other_length = self.dot(other)
        return other*(projected_length_times_other_length/other_length_sqrd)

    def cross(self, other):
        return self.x*other[1] - self.y*other[0]

    def interpolate_to(self, other, range):
        return Vector(self.x + (other[0] - self.x)*range, self.y + (other[1] - self.y)*range)

    def convert_to_basis(self, x_vector, y_vector):
        return Vector(self.dot(x_vector)/x_vector.get_length_sqrd(), self.dot(y_vector)/y_vector.get_length_sqrd())

    def __getstate__(self):
        return [self.x, self.y]

    def __setstate__(self, dict):
        self.x, self.y = dict

    def copy(self):
        return Vector(self)

    def floor(self):
        self.x = math.floor(self.x)
        self.y = math.floor(self.y)
        return self

    def ceil(self):
        self.x = math.ceil(self.x)
        self.y = math.ceil(self.y)
        return self


def main():
    pass

if __name__ == '__main__':
    main()
