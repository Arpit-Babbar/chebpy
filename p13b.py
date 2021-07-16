# solving uxx = f with zero boundary conditions for particular RHS

# Imports
from chebPy import *
from numpy import dot,zeros,sin,cos,pi,max,linspace,polyval,polyfit,inf
from numpy.linalg import norm
from scipy.linalg import solve
import matplotlib.pyplot as plt
import argparse

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument('-dir', type=str, help='Save Figure Directory', default=None)
parser.add_argument('-filename', type=str, help='File name', default=None)
args = parser.parse_args()

# Setup rhs and exact solution
def rhs(x, k):
  return (2.0*k*pi)**2 * sin(2.0*k*pi*x) + ((2.0*k+0.5)*pi)**2*cos((2.0*k+0.5)*pi*x)
def uexact(x, k):
  return -sin(2.0*k*pi*x)-cos((2.0*k+0.5)*pi*x)
# Solver
def solver(N, k):
   D,x = cheb(N)
   D2 = dot(D,D)
   D2 = D2[1:N,1:N]
   f = rhs(x[1:N],k)
   u = solve(D2,f)
   s = zeros(N+1)
   s[1:N] = u

   xx = linspace(-1.0,1.0,200)
   uu = polyval(polyfit(x,s,N),xx)    # interpolate grid data
   exact = uexact(xx,k)
   maxerr = norm(uu-exact,inf)
   fig = plt.figure()
   plt.title('$u'+' = a_0T_0+a_1T_1+a_2T_2+\\dots+a_{'+
            str(N-1)+'}T_{'+str(N-1)+'}$', fontsize=18)
   plt.plot(xx,uu,'o',xx,exact,)
   plt.legend(('approximate','exact'))
   return fig

for i in range(4):
   N, frequency = 15, 4.0
   fig = solver(15+i*5,4.0)
   if args.dir != None:
      assert args.filename!=None, "Specify file name"
      if args.dir[-1]=='/':
         fig.savefig(args.dir+args.filename+'_'+str(i)+'.pdf')
      else:
         fig.savefig(args.dir+'/'+args.filename+'_'+str(i)+'.pdf')
fig.show()