import chaospy as cp
import pylab as pl
import numpy as np

from uncertainpy import prettyPlot

Nt = 10**2
N = 50

def E_analytical(x):
    return 90*(1-np.exp(-0.1*x))/(x)


def V_analytical(x):
    return 1220*(1-np.exp(-0.2*x))/(3*x) - (90*(1-np.exp(-0.1*x))/(x))**2


def u(x, a, I):
    return I*np.exp(-a*x)


a = cp.Uniform(0, 0.1)
I = cp.Uniform(8, 10)
dist = cp.J(a, I)
T = np.linspace(0, 10, Nt+1)[1:]
dt = 10./Nt



def MC():
    samples_a = a.sample(N)
    samples_I = I.sample(N)

    U = [u(T, q, i) for q, i in zip(samples_a.T, samples_I.T)]
    U = np.array(U)

    E = (np.cumsum(U, 0).T/np.arange(1, N+1)).T
    V = (np.cumsum((U-E)**2, 0).T/np.arange(1, N+1)).T

    error = []
    var = []
    for n in xrange(N):
        error.append(dt*np.sum(np.abs(E_analytical(T) - E[n, :])))
        var.append(dt*np.sum(np.abs(V_analytical(T) - V[n, :])))

    return np.array(error), np.array(var)


def qMC():
    samples_a = a.sample(N, rule="L")
    samples_I = I.sample(N, rule="L")

    U = [u(T, q, i) for q, i in zip(samples_a.T, samples_I.T)]
    U = np.array(U)

    E = (np.cumsum(U, 0).T/np.arange(1, N+1)).T
    V = (np.cumsum((U-E)**2, 0).T/np.arange(1, N+1)).T



    error = []
    var = []
    for n in xrange(N):
        error.append(dt*np.sum(np.abs(E_analytical(T) - E[n, :])))
        var.append(dt*np.sum(np.abs(V_analytical(T) - V[n, :])))

    return np.array(error), np.array(var)



reruns = 10**2
totalerrorMC = np.zeros(N)
totalvarianceMC = np.zeros(N)
for i in xrange(reruns):
    errorMC, varianceMC = MC()

    totalerrorMC = np.add(totalerrorMC, errorMC)
    totalvarianceMC = np.add(totalvarianceMC, varianceMC)


totalerrorMC = np.divide(totalerrorMC, reruns)
totalvarianceMC = np.divide(totalvarianceMC, reruns)


reruns = 10**2
totalerrorqMC = np.zeros(N)
totalvarianceqMC = np.zeros(N)
for i in xrange(reruns):
    errorqMC, varianceqMC = qMC()

    totalerrorqMC = np.add(totalerrorqMC, errorqMC)
    totalvarianceqMC = np.add(totalvarianceqMC, varianceqMC)


totalerrorqMC = np.divide(totalerrorqMC, reruns)
totalvarianceqMC = np.divide(totalvarianceqMC, reruns)



errorCP = []
varCP = []

K = []

N = 5
for n in xrange(0, N+1):
    P = cp.orth_ttr(n, dist)
    nodes, weights = cp.generate_quadrature(n+1, dist, rule="G")
    K.append(len(nodes[0]))
    i1, i2 = np.mgrid[:len(weights), :Nt]
    solves = u(T[i2], nodes[0][i1], nodes[1][i1])

    U_hat = cp.fit_quadrature(P, nodes, weights, solves)
    errorCP.append(dt*np.sum(np.abs(E_analytical(T) - cp.E(U_hat, dist))))
    varCP.append(dt*np.sum(np.abs(V_analytical(T) - cp.Var(U_hat, dist))))



ax, tableau20 = prettyPlot(totalerrorMC[:], new_figure=True)
prettyPlot(totalvarianceMC[:], title="", xlabel="No of forward model evaluations", ylabel="Error", color=0, linestyle="--", new_figure=False)

pl.xlim([0, 49])
pl.ylim([10**-0, 2*10**1])


pl.yscale('log')
pl.legend(["Mean,      Monte Carlo", "Variance, Monte Carlo"], loc=1, prop={"size": 18})
pl.savefig("MC_convergence_2D.png")

pl.show()

prettyPlot(totalerrorMC[:], new_figure=True)
prettyPlot(totalvarianceMC[:], title="", xlabel="No of forward model evaluations", ylabel="Error", color=0, linestyle="--", new_figure=False)

prettyPlot(totalerrorqMC[:], title="", xlabel="No of forward model evaluations", ylabel="Error", color=2, new_figure=False)
prettyPlot(totalvarianceqMC[:], title="", xlabel="No of forward model evaluations", ylabel="Error", color=2, linestyle="--", new_figure=False)

pl.xlim([0, 49])
pl.ylim([10**-2, 2*10**1])

pl.yscale('log')
pl.legend(["Mean,       Monte Carlo",
           "Variance,  Monte Carlo",
           "Mean,       quasi-Monte Carlo",
           "Variance,  quasi-Monte Carlo"], loc=3, prop={"size": 18})

pl.savefig("qMC-MC_convergence_2D.png")
pl.show()


ax, tableau20 = prettyPlot(totalerrorMC[:], new_figure=True)
prettyPlot(totalvarianceMC[:], title="", xlabel="No of forward model evaluations", ylabel="Error", color=0, linestyle="--", new_figure=False)


prettyPlot(totalerrorqMC[:], title="", xlabel="No of forward model evaluations", ylabel="Error", color=2, new_figure=False)
prettyPlot(totalvarianceqMC[:], title="", xlabel="No of forward model evaluations", ylabel="Error", color=2, linestyle="--", new_figure=False)

prettyPlot(K, errorCP, title="", xlabel="No of forward model evaluations", ylabel="Error", color=4, new_figure=False)
prettyPlot(K, varCP, title="", xlabel="No of forward model evaluations", ylabel="Error", color=4, linestyle="--",new_figure=False)

pl.xlim([0, 49])
pl.ylim([10**-16, 2*10**1])

pl.yscale('log')
pl.legend(["Mean,       Monte Carlo",
           "Variance,  Monte Carlo",
           "Mean,       quasi-Monte Carlo",
           "Variance,  quasi-Monte Carlo",
           "Mean,       Polynomial Chaos",
           "Variance,  Polynomial Chaos"], loc=3, prop={"size": 16})

pl.savefig("qMC-MC-PC_convergence_2D.png")


pl.show()
