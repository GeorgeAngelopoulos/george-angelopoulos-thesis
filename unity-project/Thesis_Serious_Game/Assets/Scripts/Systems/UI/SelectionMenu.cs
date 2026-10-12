using UnityEngine;
using TMPro;
using System;
using UnityEngine.InputSystem;
using UnityEngine.UI;

public class SelectionMenu : MonoBehaviour
{
    public HexNode selectedNode;
    public GameObject selectedObject;

    public enum SelectState {Hex, Predator, Prey};

    private float exploreCost;

    #region UI Text References

    [Header("UI Elements Reference")]
    [SerializeField] private TMP_Text ContentTitleText;
    [SerializeField] private Image contentBG;
    [SerializeField] private GameObject SelectedHexInfo;
    [SerializeField] private GameObject SelectedPredatorInfo;
    [SerializeField] private GameObject SelectedPreyInfo;

    [Header("UI Hex Text References")]
    [SerializeField] private TMP_Text SelectHexQText;
    [SerializeField] private TMP_Text SelectHexRText;
    [SerializeField] private TMP_Text selectHexWalkableText;
    [SerializeField] private TMP_Text selectedHexCostText;
    [SerializeField] private Button exploreButton;

    [Header("UI Predator Text References")]
    [SerializeField] private TMP_Text SelectPredatorMeritText;
    [SerializeField] private TMP_Text SelectPredatorDemandText;
    [SerializeField] private TMP_Text SelectPredatorEffortText;
    [SerializeField] private TMP_Text SelectPredatorTacticText;
    [SerializeField] private TMP_Text SelectPredatorFoodText;
    [SerializeField] private Button recallButton;  

    [Header("UI Prey Text References")]
    [SerializeField] private TMP_Text SelectPreyPsoloText;
    [SerializeField] private TMP_Text SelectPreyRewardText;
    [SerializeField] private TMP_Text SelectPreyBetaText;

    #endregion

    public void ShowSelectContent(SelectState activeState)
    {
        foreach (Transform child in transform)
        {
            child.gameObject.SetActive(false);
        }

        contentBG.gameObject.SetActive(true);
        switch (activeState)
        {
            case SelectState.Hex:
                ContentTitleText.gameObject.SetActive(true);
                SelectedHexInfo.gameObject.SetActive(true);
                selectedHexCostText.gameObject.SetActive(true);
                exploreButton.gameObject.SetActive(true);
                UpdateSelectedHexUI();
                break;

            case SelectState.Predator:
                ContentTitleText.gameObject.SetActive(true);
                SelectedPredatorInfo.gameObject.SetActive(true);
                recallButton.gameObject.SetActive(true);
                UpdateSelectedPredatorUI();
                break;

            case SelectState.Prey:
                ContentTitleText.gameObject.SetActive(true);
                SelectedPreyInfo.gameObject.SetActive(true);
                UpdateSelectedPreyUI();
                break;
        }
    }

    #region UI Hex Functions

    public void onHexExplorePressed()
    {
        HexGridManager.Instance.SetNodeWalkable(selectedNode.q, selectedNode.r, true);
        HexGridManager.Instance.meshGenerator.UpdateMeshColors(HexGridManager.Instance.nodes.ToArray());
        GameManager.Instance.SpendFood(exploreCost);
        SpawnManager.Instance.SpawnPreyGroupInHex(selectedNode.q, selectedNode.r);
        UIManager.Instance.onChangeDashboardState(0);
    }
    public void UpdateSelectedHexUI()
    {

        ContentTitleText.text = "Hex Exploration";

        SelectHexQText.text = $"Q: {selectedNode.q}";
        SelectHexRText.text = $"R: {selectedNode.r}";
        
        if (selectedNode.isWalkable) 
        { 
            selectHexWalkableText.text = "<Color=green> Already Explored </color>";

            selectedHexCostText.gameObject.SetActive(false);
            exploreButton.gameObject.SetActive(false);
        }
        else 
        { 
            selectHexWalkableText.text = "<Color=red> Not Explored </color>";

            exploreCost = calculateExploreCost(selectedNode);
            selectedHexCostText.text = $"Explore Cost: {exploreCost} food";

            selectedHexCostText.gameObject.SetActive(true);
            exploreButton.gameObject.SetActive(true);  
        }

    }
    private float calculateExploreCost(HexNode node)
    {
        int distanceFromCenter = (Math.Abs(node.q) + Math.Abs(node.r) + Math.Abs(node.q + node.r)) / 2;
        float calculatedCost = distanceFromCenter * SimulationConfig.C_HEX_EXPLORE;
        return calculatedCost;   
    }

    #endregion

    #region UI Predator Functions

    public void onRecallPredatorPressed()
    {
        
    }

    public void UpdateSelectedPredatorUI()
    {
        

        Predator predatorStats = selectedObject.GetComponent<Predator>();

        if (predatorStats != null)
        {
            ContentTitleText.text = $"{predatorStats.agentType.ToString()} Predator";

            SelectPredatorMeritText.text = $"Merit: {predatorStats.merit}";
            SelectPredatorDemandText.text = $"Base Demand: {predatorStats.baseDemand}";
            SelectPredatorEffortText.text = $"Hunt Cost: {predatorStats.huntCost} fd";
            switch (predatorStats.tactic)
            {
                case "Egalitarian":
                    SelectPredatorTacticText.text = $"<Color=green> Egalitarian </color>";
                    break;
                case "Meritocratic":
                    SelectPredatorTacticText.text = $"<Color=blue> Meritocratic </color>";
                    break;
                case "Altruistic":
                    SelectPredatorTacticText.text = $"<Color=red> Altruistic </color>";
                    break;
                case "Asymmetric":
                    SelectPredatorTacticText.text = $"<Color=yellow> Asymmetric </color>";
                    break;
            }
        }
    }

    #endregion

    #region UI Prey Functions

    public void UpdateSelectedPreyUI()
    {
        Prey preyStats = selectedObject.GetComponent<Prey>();

        if (preyStats != null)
        {
            ContentTitleText.text = $"{preyStats.preyType.ToString()}";

            SelectPreyPsoloText.text = $"Solo Probability: {preyStats.pSolo*100} %";
            SelectPreyRewardText.text = $"Reward: {preyStats.reward}";
            SelectPreyBetaText.text = $"Synergy Bonus: {preyStats.beta}";
        }
    }

    #endregion

    
    

}
