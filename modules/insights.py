class InsightGenerator:

    @staticmethod
    def generate(df):
        insights = []

        most_nodes = df.loc[df["nodes"].idxmax(), "dataset"]
        most_edges = df.loc[df["edges"].idxmax(), "dataset"]
        densest = df.loc[df["density"].idxmax(), "dataset"]

        insights.append(f"📌 {most_nodes} has the highest number of nodes.")
        insights.append(f"📌 {most_edges} is the most connected network.")
        insights.append(f"📌 {densest} is the densest graph.")

        return insights