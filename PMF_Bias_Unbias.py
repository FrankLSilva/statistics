# Pip install thinkx mandatory!
import thinkstats2
import thinkplot

# PROBABILITY MASS FUNCTIONS

# ------------- d = Dictionary Histogram values
d = {7: 8, 12: 8, 17: 14, 22: 4, 27: 6, 32: 12, 37: 8, 42: 3, 47: 2}
pmf = thinkstats2.Pmf(d, label="actual")

# ------------- BIAs / UNBIAS functions
def BiasPmf(pmf, label):
    new_pmf = pmf.Copy(label=label)

    for x, p in pmf.Items():
        new_pmf.Mult(x, x)
    new_pmf.Normalize()
    return new_pmf

def UnbiasPmf(pmf, label=None):
    new_pmf = pmf.Copy(label=label)

    for x, p in pmf.Items():
        new_pmf[x] *= 1 / x

    new_pmf.Normalize()
    return new_pmf

ploting = True
while ploting:
    choose = int(input("\nSelect PLOT:\n"
                       "1. BIAS PMF\n"
                       "2. UNBIAS PMF\n"))

    # ------------- BIASED PMFs
    if choose == 1:
        biased_pmf = BiasPmf(pmf, label="observed")
        print("\nActual mean", pmf.Mean())
        print("Observed mean", biased_pmf.Mean())
        print("------------------------")

        thinkplot.PrePlot(2)
        thinkplot.Pmfs([pmf, biased_pmf])
        thinkplot.Config(xlabel="Class size", ylabel="PMF")
        thinkplot.show()

    # ------------- UNBIASED PMFs
    elif choose == 2:
        unbiased = UnbiasPmf(biased_pmf, label="unbiased")
        print("Unbiased mean", unbiased.Mean())
        print("------------------------")

        thinkplot.PrePlot(2)
        thinkplot.Pmfs([pmf, unbiased])
        thinkplot.Config(xlabel="Class size", ylabel="PMF")
        thinkplot.show()

    else:
        ploting = False