from tf_transformations import quaternion_matrix

def transform_to_matrix(transform_stamped):
    t = transform_stamped.transform.translation
    q = transform_stamped.transform.rotation
    matrix = quaternion_matrix([q.x, q.y, q.z, q.w])
    matrix[0, 3] = t.x
    matrix[1, 3] = t.y
    matrix[2, 3] = t.z
    return matrix