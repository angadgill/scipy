---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"tags": ["jupyterlite_sphinx_strip"]}

```{eval-rst}
.. notebooklite:: hypothesis_ztest.md
   :new_tab: True
```

(hypothesis_ztest)=

+++

# Z-tests

A z-test is a hypothesis test in which the test statistic is compared
against the standard normal distribution. It is typically used for means
when the population standard deviation $\sigma$ is known, and for proportions
when the sample is large.

SciPy does not provide a function for z-test directly. In most situations
where a z-test is used, SciPy either offers a test that makes weaker
assumptions (for example, not requiring the population variance to be known,
or not relying on a large-sample approximation), or an existing function
already computes the same p-value. If absolutely needed, the z-test can be
written with only a few lines of code using {class}`scipy.stats.norm`. In
this tutorial, we identify which tests apply when and show how to perform
them.

## Which test to use

For tests of **means**:

| Samples | Population variance | Use |
|---|---|---|
| One | Known | {class}`~scipy.stats.norm` directly ({ref}`below <ztest_one_known>`) |
| Two independent | Known | {class}`~scipy.stats.norm` directly ({ref}`below <ztest_two_known>`) |
| One | Estimated from the data | {func}`~scipy.stats.ttest_1samp` |
| Paired | Estimated from the data | {func}`~scipy.stats.ttest_rel` |
| Two independent | Estimated from the data | {func}`~scipy.stats.ttest_ind`, {func}`~scipy.stats.ttest_ind_from_stats` |

For tests of **proportions**:

| Samples | Use |
|---|---|
| One | {func}`~scipy.stats.binomtest` |
| Two independent | {func}`~scipy.stats.chi2_contingency` with `correction=False`; {func}`~scipy.stats.barnard_exact` or {func}`~scipy.stats.boschloo_exact` when counts are small |

## Means with known variance

A variance is known when it is not estimated from the samples being
tested; for example, when it is a property of a measurement process that
has been characterized in advance. When the data are normally distributed,
the z statistic follows the standard normal distribution exactly, at any
sample size. When they are not, the central limit theorem makes the
standard normal an approximation that improves as the sample size grows.

(ztest_one_known)=

### One mean

Montgomery and Runger [^1] describe a solid propellant used in air crew
escape systems. Specifications require that the mean burning rate be
50 centimeters per second, and the standard deviation of the burning rate
is known to be 2 centimeters per second. A random sample of 25 specimens
has a mean burning rate of 51.3 centimeters per second.

We begin by formulating a null hypothesis $H_0$:

> The mean burning rate is 50 centimeters per second.

The standard deviation is a known property of the process, not an estimate
from the sample of 25 specimens, and we rely on burning rates being
normally distributed. Under the null hypothesis, the statistic

$$
z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}
$$

follows the standard normal distribution, where $\bar{x}$ is the sample
mean, $\mu_0$ is the hypothesized mean, and $n$ is the sample size.

```{code-cell}
import numpy as np
from scipy import stats

x_bar, mu_0, sigma, n = 51.3, 50, 2, 25
z = (x_bar - mu_0) / (sigma / np.sqrt(n))
z
```

The two-sided p-value is the probability of a statistic at least this
extreme in either direction.

```{code-cell}
2 * stats.norm.sf(abs(z))
```

Using a significance level of 5%, we would reject the null hypothesis in
favor of the alternative hypothesis: "The mean burning rate differs from
50 centimeters per second."

(ztest_two_known)=

### Two means

Montgomery and Runger [^1] also describe an experiment comparing the drying
time of two formulations of a primer paint. Experience with the process
indicates that the standard deviation of drying time is 8 minutes, and that
the new formulation does not affect it. Ten specimens painted with the old
formulation have a mean drying time of 121 minutes; ten specimens painted
with the new formulation have a mean drying time of 112 minutes.

We begin by formulating a null hypothesis $H_0$:

> The mean drying time of the new formulation is the same as that of the old
> formulation.

With only ten specimens per formulation, we rely on drying times being
normally distributed. The statistic is the difference in sample means
divided by its standard error:

$$
z = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\sigma_1^2 / n_1 + \sigma_2^2 / n_2}}
$$

```{code-cell}
import numpy as np
from scipy import stats

x_bar1, sigma1, n1 = 121, 8, 10  # old formulation
x_bar2, sigma2, n2 = 112, 8, 10  # new formulation
z = (x_bar1 - x_bar2) / np.sqrt(sigma1**2 / n1 + sigma2**2 / n2)
z
```

We are interested in whether the new formulation reduces the drying time, so
we compute a one-sided p-value.

```{code-cell}
stats.norm.sf(z)
```

Using a significance level of 5%, we would reject the null hypothesis in
favor of the alternative hypothesis: "The mean drying time of the new
formulation is less than that of the old formulation."

## Means with unknown variance: the t-test

In most applications, $\sigma$ is not known and must be estimated from the
data. Replacing $\sigma$ with the sample standard deviation $s$ changes the
null distribution of the statistic: when the data are normally distributed,
it follows Student's t distribution rather than the standard normal. The
t-test is the exact test for this situation; it is not an approximation to
the z-test. As the sample size grows, the t distribution approaches the
standard normal and the two p-values converge, but at no sample size is the
z-test the more accurate of the two. When the data are not normally
distributed, neither test is exact. Because the t distribution has heavier
tails than the standard normal, the t-test always gives the larger, more
conservative p-value for the same statistic.

### One sample

Montgomery and Runger [^1] report the coefficient of restitution, the ratio
of a golf ball's outgoing to incoming velocity, of 15 drivers produced by a
club maker.

```{code-cell}
import numpy as np
from scipy import stats

restitution = np.array([0.8411, 0.8191, 0.8182, 0.8125, 0.8750,
                        0.8580, 0.8532, 0.8483, 0.8276, 0.7983,
                        0.8042, 0.8730, 0.8282, 0.8359, 0.8660])
```

We begin by formulating a null hypothesis $H_0$:

> The mean coefficient of restitution is 0.82.

We are interested in whether the mean exceeds 0.82, so let's assess the
plausibility of this hypothesis with a one-sided
{func}`one-sample t-test <scipy.stats.ttest_1samp>`.

```{code-cell}
res = stats.ttest_1samp(restitution, popmean=0.82, alternative='greater')
res.statistic
```

```{code-cell}
res.pvalue
```

Using a significance level of 5%, we would reject the null hypothesis in
favor of the alternative hypothesis: "The mean coefficient of restitution
exceeds 0.82."

Had we compared the same statistic against the standard normal
distribution, as a z-test with estimated variance would, the p-value would
be less than half as large.

```{code-cell}
stats.norm.sf(res.statistic)
```

### Paired samples

Montgomery and Runger [^1] compare two methods for predicting the shear
strength of steel plate girders, the Karlsruhe and Lehigh methods, applied
to the same nine girders. Each value is the ratio of predicted to observed
load.

```{code-cell}
import numpy as np
from scipy import stats

karlsruhe = np.array([1.186, 1.151, 1.322, 1.339, 1.200,
                      1.402, 1.365, 1.537, 1.559])
lehigh = np.array([1.061, 0.992, 1.063, 1.062, 1.065,
                   1.178, 1.037, 1.086, 1.052])
```

Both methods were applied to each girder, so the measurements are paired.
Our null hypothesis is:

> The mean prediction is the same for both methods.

We perform a {func}`paired t-test <scipy.stats.ttest_rel>`.

```{code-cell}
res = stats.ttest_rel(karlsruhe, lehigh)
res.pvalue
```

Using a significance level of 5%, we would reject the null hypothesis in
favor of the alternative hypothesis: "The mean prediction differs between
the two methods." This is equivalent to applying
{func}`~scipy.stats.ttest_1samp` to the differences `karlsruhe - lehigh`.

### Two independent samples

For two independent samples, {func}`~scipy.stats.ttest_ind` performs
Student's t-test by default, which assumes that the two populations have
the same variance. With `equal_var=False`, it performs Welch's t-test,
which does not make this assumption. Student's t-test is sensitive to the
assumption of equal variances, particularly when the sample sizes differ.
Unless there is good reason to believe that the variances are equal, Welch's
t-test is the safer choice.

Montgomery and Runger [^1] report arsenic concentrations in the drinking
water of 10 communities in metropolitan Phoenix and 10 communities in rural
Arizona, in parts per billion. They note that a normal probability plot of
the two samples suggests that the population variances differ.

```{code-cell}
import numpy as np
from scipy import stats

metro = np.array([3, 7, 25, 10, 15, 6, 12, 25, 15, 7])
rural = np.array([48, 44, 40, 38, 33, 21, 20, 12, 1, 18])
```

Our null hypothesis is:

> The mean arsenic concentration is the same in metropolitan Phoenix and
> rural Arizona.

```{code-cell}
res = stats.ttest_ind(metro, rural, equal_var=False)
res.pvalue
```

Using a significance level of 5%, we would reject the null hypothesis in
favor of the alternative hypothesis: "The mean arsenic concentration
differs between metropolitan Phoenix and rural Arizona."

When only summary statistics are available,
{func}`~scipy.stats.ttest_ind_from_stats` accepts the means, standard
deviations, and sample sizes instead of the data.

## Proportions

Z-tests for proportions rely on the normal approximation to the binomial
distribution, which is accurate only when the sample is large. SciPy offers
tests that compute p-values from the discrete distributions directly, and a
chi-square test that is identical to the two-proportion z-test.

### One proportion

{func}`~scipy.stats.binomtest` computes the p-value from the binomial
distribution itself, so it does not depend on the normal approximation.

Suppose that a supplier states that 5% of its parts are defective, and an
inspection of 300 parts finds 28 defective. Our null hypothesis is:

> The proportion of defective parts is 5%.

```{code-cell}
from scipy import stats

res = stats.binomtest(k=28, n=300, p=0.05)
res.pvalue
```

Using a significance level of 5%, we would reject the null hypothesis in
favor of the alternative hypothesis: "The proportion of defective parts
differs from 5%."

For comparison, the z-test p-value is several times smaller than the exact
p-value, because the normal approximation is poor in the tail of a
binomial distribution with a small success probability.

```{code-cell}
import numpy as np
from scipy import stats

k, n, p_0 = 28, 300, 0.05
z = (k / n - p_0) / np.sqrt(p_0 * (1 - p_0) / n)
2 * stats.norm.sf(abs(z))
```

### Two proportions

The two-proportion z-test is the usual way of analyzing a binary outcome in
an A/B test. Suppose that a website shows a blue button to 30,000 visitors
and a red button to another 30,000 visitors, and records whether each
visitor clicks:

|             | Clicked | Did not click |
|-------------|---------|---------------|
| Blue button |  1,500  |    28,500     |
| Red button  |  1,650  |    28,350     |

We begin by formulating a null hypothesis $H_0$:

> The probability that a visitor clicks is the same for both buttons.

The pooled two-proportion z-test is algebraically identical to the
chi-square test of independence on this table without continuity
correction: the chi-square statistic is the square of the z statistic, and
the p-values are equal. Let's perform the test with
{func}`~scipy.stats.chi2_contingency`.

```{code-cell}
from scipy import stats

table = [[1500, 28500], [1650, 28350]]
res = stats.chi2_contingency(table, correction=False)
res.pvalue
```

To confirm the equivalence, we compute the z statistic directly.

```{code-cell}
import numpy as np
from scipy import stats

k1, n1 = 1500, 30000  # blue button
k2, n2 = 1650, 30000  # red button
p1, p2 = k1 / n1, k2 / n2
p = (k1 + k2) / (n1 + n2)  # pooled proportion
z = (p2 - p1) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
2 * stats.norm.sf(abs(z))
```

Using a significance level of 5%, we would reject the null hypothesis in
favor of the alternative hypothesis: "The probability that a visitor clicks
differs between the two buttons."

The `correction=False` argument is required. By default,
{func}`~scipy.stats.chi2_contingency` applies Yates' continuity correction
to 2x2 tables: it reduces the absolute difference between each observed and
expected count by 0.5 before computing the statistic. The correction is
intended to compensate for approximating a discrete distribution with the
continuous chi-square distribution, and it makes the test more
conservative. The z-test does not include it, so the default p-value is
larger than the z-test p-value.

```{code-cell}
from scipy import stats

table = [[1500, 28500], [1650, 28350]]
stats.chi2_contingency(table).pvalue
```

When counts are small, the normal approximation that underlies both the
z-test and the chi-square test becomes unreliable.
{func}`~scipy.stats.barnard_exact` and {func}`~scipy.stats.boschloo_exact`
compute p-values without relying on it. Consider the acetazolamide data
from the {ref}`Fisher's exact test tutorial <hypothesis_fisher_exact>`, in
which 7 of 22 participants who took acetazolamide and 17 of 22 participants
who took a placebo developed acute mountain sickness.

```{code-cell}
from scipy import stats

table = [[7, 15], [17, 5]]
stats.chi2_contingency(table, correction=False).pvalue
```

```{code-cell}
stats.barnard_exact(table).pvalue
```

```{code-cell}
stats.boschloo_exact(table).pvalue
```

Here the conclusions agree, but only the exact tests guarantee that the
probability of a false rejection does not exceed the significance level.
As discussed in the {ref}`Fisher's exact test tutorial
<hypothesis_fisher_exact>`, Boschloo's test is more powerful than Fisher's
exact test when the row sums are not fixed by the design of the experiment.

## References

[^1]: Montgomery, Douglas C. and George C. Runger (2018) *Applied
Statistics and Probability for Engineers*, 7th ed. Hoboken: Wiley.
