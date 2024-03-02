import pandas as pd
import numpy as np
from munkres import Munkres
from .constrained_kmedoids import KMedoids

def group_mentors(mentors: pd.DataFrame,
                  mentors_per_mentee: int,
                  similarity_func: callable):
    '''KMedoids constrained clustering to group mentors based on similarity.
    
    Args:
        mentors: pd.DataFrame, representing the mentors
        mentors_per_mentee: int, the number of mentors per mentee
        similarity_func: callable, a function that takes two pd.Series and returns a number. Smaller is more similar.
    
    Returns:
        dict, mapping mentor group IDs to lists of mentor IDs
    '''
    # Generate similarity matrix
    similarity_matrix = pd.DataFrame(index=mentors.index, columns=mentors.index)
    for mentor_id1 in mentors.index:
        for mentor_id2 in mentors.index:
            if mentor_id1 == mentor_id2:
                similarity_matrix.loc[mentor_id1, mentor_id2] = np.inf
                continue
            similarity_matrix.loc[mentor_id1, mentor_id2] = similarity_func(mentors.loc[mentor_id1], mentors.loc[mentor_id2])
    
    # Cluster mentors
    n_clusters = len(mentors.index) // mentors_per_mentee
    km = KMedoids(distance_matrix=similarity_matrix.values, n_clusters=n_clusters)
    km.run(max_iterations=10, tolerance=0.001)

    return km.clusters

def match_mentees_to_mentor_groups(mentors: pd.DataFrame, 
                             mentees: pd.DataFrame, 
                             mentor_groups: dict,
                             mentees_per_mentor: int, 
                             similarity_func: callable):
    '''Modreg-style matching of mentees to mentor groups using the Hungarian method.

    Both mentor and mentee POV are returned for convenience.
    
    Args:
        mentors: pd.DataFrame, representing the mentors
        mentees: pd.DataFrame, representing the mentees
        mentor_groups: dict, mapping mentor group IDs to lists of mentor IDs
        mentees_per_mentor: int, the number of mentees per mentor
        similarity_func: callable, a function that takes two pd.Series and returns a number. Smaller is more similar.
        
    Returns:
        pd.DataFrame, representing the assignments from mentor POV.
        pd.DataFrame, representing the assignments by mentee POV.
    '''
    # Generate similarity matrix
    similarity_matrix = pd.DataFrame(index=mentor_groups.keys(), columns=mentees.index)
    for mentor_group_id, mentor_group in mentor_groups.items():
        for mentee_id, mentee in mentees.iterrows():
            similarity_matrix.loc[mentor_group_id, mentee_id] = similarity_func([mentors.iloc[mentor_id] for mentor_id in mentor_group], mentee)

    assignments = pd.DataFrame(index=mentor_groups.keys(), columns=[f'assigned_{i}' for i in range(mentees_per_mentor)])

    # Match mentees to mentor groups
    mentees_pool = mentees.copy()
    for round in range(mentees_per_mentor):
        matchings = Munkres().compute(similarity_matrix.values.astype(np.float32))
        for mentor_group_id_index, mentee_id_index in matchings:
            matched_mentee = mentees_pool.index[mentee_id_index]
            matched_mentor_group = list(mentor_groups.keys())[mentor_group_id_index]

            assignments.loc[matched_mentor_group][f'assigned_{round}'] = matched_mentee

        similarity_matrix = similarity_matrix.drop(assignments[f'assigned_{round}'], axis=1)
        mentees_pool = mentees_pool.drop(assignments[f'assigned_{round}'])

    # Generate table of assignments from mentor POV for convenience
    assignments_by_mentor = pd.DataFrame(
        index=mentors.index, 
        columns=[f'assignment_{i}' for i in range(mentees_per_mentor)])

    for assignment in assignments.iterrows():
        mentor_group = mentor_groups[assignment[0]]

        for mentor in mentor_group:
            assignments_by_mentor.iloc[mentor] = assignment[1]

    # Generate table of assignments from mentee POV for convenience
    max_mentor_group_size = max(len(mentor_group) for mentor_group in mentor_groups.values())
    assignments_by_mentee = pd.DataFrame(index=mentees.index, columns=[f'Mentor {i}' for i in range(max_mentor_group_size)])


    for round, value in assignments.items():
        for mentor_group, mentee in value.items():
            mentor_group_names = [mentors.iloc[mentor_id].name for mentor_id in mentor_groups[mentor_group]]

            # Pad with NaNs to fit with assignments_by_mentee dimensions
            if len(mentor_group_names) < max_mentor_group_size:
                mentor_group_names += [np.nan] * (max_mentor_group_size - len(mentor_group_names))
            assignments_by_mentee.loc[mentee] = mentor_group_names

    return assignments_by_mentor, assignments_by_mentee

def match(mentors: pd.DataFrame, 
          mentees: pd.DataFrame, 
          mentors_per_mentee: int, 
          mentees_per_mentor: int, 
          similarity_mentee_mentor_group: callable, 
          similarity_mentor_mentor: callable):
    ''' Match mentees to mentors using a two-step process: 
    1. Group mentors together into mentor groups
    2. Match mentees to mentor groups
    
    Args:
        mentors: pd.DataFrame, representing the mentors
        mentees: pd.DataFrame, representing the mentees
        mentors_per_mentee: int, the number of mentors per mentee
        mentees_per_mentor: int, the number of mentees per mentor
        similarity_mentee_mentor_group: callable, a function that takes a list of pd.Series and a pd.Series and returns a number. Smaller is more similar.
        similarity_mentor_mentor: callable, a function that takes two pd.Series and returns a number. Smaller is more similar.
        
    Returns:
        pd.DataFrame, representing the assignments from mentor POV.
        pd.DataFrame, representing the assignments by mentee POV.
    '''
    groups = group_mentors(mentors, mentors_per_mentee, similarity_mentor_mentor)
    assignments_by_mentor, assignments_by_mentee = match_mentees_to_mentor_groups(mentors, mentees, groups, mentees_per_mentor, similarity_mentee_mentor_group)
    return assignments_by_mentor, assignments_by_mentee
