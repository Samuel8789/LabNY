# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 20:51:11 2024

@author: sp3660
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
movie_corr=[0.362302,
0.181071,
0.480991,
0.256597,
0.413265,
0.356984,
0.215649,
0.4798565,
0.336598,

]

spont_corr=[0.307302,
0.195046,
0.524189,
0.323598,
0.222356,
0.389365,
0.406359,
0.319856,
0.266893,

]
matplotlib.rcParams.update({'font.size': 40})
# Create a DataFrame
df = pd.DataFrame({
    'Value': movie_corr + spont_corr,
    'Condition': ['Natural Movie'] * len(movie_corr) + ['Spontaneous'] * len(spont_corr)
})

# Create a boxplot with individual points
f,ax=plt.subplots(figsize=(5, 15))
fuptit='Correlation Across Cells'


# Boxplot
sns.boxplot(x='Condition', y='Value', data=df, color='lightgray', showfliers=False)

# Overlay individual data points
sns.stripplot(x='Condition', y='Value', data=df, jitter=True, color='black', alpha=0.7)

# Add labels and title
ax.set_ylabel('Correlation Coefficient')
ax.set_ylim([-1,1])

for spine in   ax.spines.values():
    if  not any(spine ==  ax.spines['bottom'] or spine== ax.spines['left']):
        spine.set_visible(False)
# Show plot
plt.show()


# plt.title(fuptit)

plt.tight_layout()
plt.savefig(temppath /  Path(fuptit+f'_{timestr}.pdf'), dpi=300, bbox_inches='tight')
#%% acros sessions
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
movie_corr=[0.426447,
0.238698,
0.130991,
0.548393,
0.400663,
0.528054,
0.520432,
0.423569,
0.385693,
0.516973,
0.36978,
0.265987,
0.372689,
]

spont_corr=[-0.142272,
-0.166264,
-0.00340466,
-0.138595,
-0.247071,
0.0153456,
-0.149067,
0.1216597,
0.054876,
-0.035996,
0.154799,
-0.226898,
-0.098956,


]
matplotlib.rcParams.update({'font.size': 40})
# Create a DataFrame
df = pd.DataFrame({
    'Value': movie_corr + spont_corr,
    'Condition': ['Natural Movie'] * len(movie_corr) + ['Spontaneous'] * len(spont_corr)
})

# Create a boxplot with individual points
f,ax=plt.subplots(figsize=(5, 15))
fuptit='Correlation Across Sessions'


# Boxplot
sns.boxplot(x='Condition', y='Value', data=df, color='lightgray', showfliers=False)

# Overlay individual data points
sns.stripplot(x='Condition', y='Value', data=df, jitter=True, color='black', alpha=0.7)

# Add labels and title
ax.set_ylabel('Correlation Coefficient')
ax.set_ylim([-1,1])

for spine in   ax.spines.values():
    if  not any(spine ==  ax.spines['bottom'] or spine== ax.spines['left']):
        spine.set_visible(False)
# Show plot
plt.show()

plt.tight_layout()
plt.savefig(temppath /  Path(fuptit+f'_{timestr}.pdf'), dpi=300, bbox_inches='tight')