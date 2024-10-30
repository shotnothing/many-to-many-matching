# Many-to-many Matching
Package to conduct matching for an extension of the stable marriage problem to many-to-many matchings.

Requirements:
- pandas
- numpy
- munkres

Example usage:
```python
import pandas as pd
import numpy as np
import manytomany

# 1. Generate some fake data
mentee_df = pd.DataFrame({
    'id': ['sA', 'sB', 'sC', 'sD', 'sE', 'sF', 'sG', 'sH', 'sI', 'sJ', 'sK', 'sL', 'sM'], 
    'feat1': [1, 2, 3, 4, 5 ,6, 7, 8, 9, 10, 11, 12, 13],
    'feat2': [5, 4, 3, 5, 4 ,3, 5, 4, 3, 6, 5, 4, 3],
}).set_index('id')

mentor_df = pd.DataFrame({
    'id': ['mA', 'mD', 'mF', 'mJ', 'mK', 'mL', 'mN'], 
    'feat1': [1, 4, 6 ,10, 11, 12, 14],
    'feat2': [1, 2, 3, 4, 5, 6, 7],
}).set_index('id')

# 2. Define similarity functions
def similarity_mentee_mentor_group(mentors: list, mentee: pd.Series):
    '''You can define any similarity function you want, as long as you return a number (you might be able to return other comparable objects but I haven't tested it). Smaller is more similar.
    
    Args:
        mentors: list of pd.Series, each representing a mentor
        mentee: pd.Series, representing a single mentee
    '''
    acc = 0
    acc += sum(
        np.abs(mentor['feat1'] - mentee['feat1'])
        for mentor in mentors
    )
    acc += sum(
        np.abs(mentor['feat2'] - mentee['feat2'])
        for mentor in mentors
    )
    return acc

def similarity_mentor_mentor(mentor1: pd.Series, mentor2: pd.Series):
    '''Again, you can define any similarity function you want, as long as you return a number. Smaller is more similar.
    
    Args:
        mentor1: pd.Series, representing a single mentor
        mentor2: pd.Series, representing a single mentor
    '''
    return np.abs(mentor1['feat1'] - mentor2['feat1']) * np.abs(mentor1['feat2'] - mentor2['feat2'])**0.15

# 3. Run the matching
assignments_by_mentor, assignments_by_mentee = manytomany.match(
    mentors = mentor_df,
    mentees = mentee_df,
    mentors_per_mentee = 2,
    mentees_per_mentor = 2,
    similarity_mentee_mentor_group = similarity_mentee_mentor_group,
    similarity_mentor_mentor = similarity_mentor_mentor
)

print('Assignments by mentor:\n', assignments_by_mentor)
print('Assignments by mentee:\n', assignments_by_mentee)
```
