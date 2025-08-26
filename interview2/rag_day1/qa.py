CONFIDENCE_T = 0.25

def compose_answer(query, hits):
    top = hits[0] if hits else None
    if not top or top["score"] < CONFIDENCE_T:
        return "I don't know.", []
    cites = [(h["doc"], round(h["score"], 2)) for h in hits[:2]]
    return f"Answer based on {cites[0][0]}. See citations.", cites

