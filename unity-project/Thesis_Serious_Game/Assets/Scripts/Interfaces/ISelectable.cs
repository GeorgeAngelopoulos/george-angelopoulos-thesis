using UnityEngine;

public interface ISelectable
{
    // Anything implementing this MUST declare what happens when clicked
    void OnClick(Vector3 hitPoint);
}
