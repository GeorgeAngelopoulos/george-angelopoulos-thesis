using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem;

public class SelectionManager : MonoBehaviour
{
    private void Update()
    {
        // 1. Listen for Left Click (New Input System)
        if (Mouse.current != null && Mouse.current.leftButton.wasPressedThisFrame)
        {
            // 2. Universal UI Block: Stop immediately if clicking a UI Element
            if (EventSystem.current != null && EventSystem.current.IsPointerOverGameObject()) 
            {
                return; 
            }

            // 3. Fire a single world raycast
            Ray ray = Camera.main.ScreenPointToRay(Mouse.current.position.ReadValue());
            if (Physics.Raycast(ray, out RaycastHit hit))
            {
                // 4. Look for ANY component on the object that implements ISelectable
                ISelectable selectable = hit.collider.GetComponent<ISelectable>();
                
                if (selectable != null) 
                { 
                    selectable.OnClick(hit.point); 
                }
            }
        }
    }

}
