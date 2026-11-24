# Business Recommendation: AI-Based Service Request Routing

Date: 24 November 2026  
Subject: Evaluation of an ML-Based Service Request Routing Solution

### 1. Executive Summary

We developed a machine-learning prototype to classify home appliance service requests into seven service teams. The solution uses TF-IDF text features and a Linear Support Vector Machine (LinearSVC), with a FastAPI backend and Streamlit employee dashboard.

On a chronological validation set of 2,165 historical requests, the model achieved **95.20% accuracy and 95.00% macro F1 score**.

### 2. Business Value

The existing routing process has opportunities for improvement. Historical resolution records show that approximately 22.83% of requests were transferred from their initial team to a different final team.

The supplied cost assumptions are:

- ₹305 per team transfer.
- ₹260 in additional customer-contact cost per misrouted request.
- ₹3.2 lakh annual routing-bot licence cost.

Based on the historical records, estimated transfer and additional-contact costs total approximately ₹18.33 lakh. This is a historical cost estimate, not a guaranteed saving from the proposed model.

The proposed solution also avoids a per-request paid AI API dependency because it uses a locally trained model.

### 3. Recommendation

I recommend a controlled pilot rather than immediately replacing the existing routing system.

The pilot should:

1. Run the ML model alongside the current routing process.
2. Compare predicted teams with actual employee routing decisions.
3. Track transfer rates and additional customer contacts.
4. Identify ambiguous requests that need employee clarification.
5. Evaluate operational cost and performance before making a replacement decision.

### 4. Important Operational Considerations

The model predicts the historical initial bot-assigned team, not necessarily the final team that resolves a request. Therefore, its validation score measures how well it reproduces historical initial routing decisions.

Requests should also follow the operations policy. For example, Billing should be selected only when the payment itself is the issue. Simply mentioning a payment should not automatically result in Billing routing.

Requests that lack sufficient information should be reviewed or clarified by an employee.

### 5. Conclusion

The prototype demonstrates that a locally hosted machine-learning solution can classify historical service requests with promising validation performance.

A controlled pilot, supported by employee review and cost monitoring, is the recommended next step before deciding whether to replace the existing routing bot.