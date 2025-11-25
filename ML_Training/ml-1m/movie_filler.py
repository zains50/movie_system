def fill_missing_ids(input_file, output_file):
    lines = {}
    ids = []

    # Read all lines and extract IDs
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("::")
            if parts and parts[0].isdigit():
                movie_id = int(parts[0])
                ids.append(movie_id)
                lines[movie_id] = line.strip()

    ids.sort()
    all_ids = range(ids[0], ids[-1] + 1)

    # with open(output_file, "w", encoding="utf-8") as f:
    #     for i in all_ids:
    #         if i in lines:
    #             f.write(lines[i] + "\n")
    #         else:
    #             f.write(f"{i}::FAKE MOVIE::Drama\n")
    #
    # print(f"✅ Output written to '{output_file}'")
    print(f"🎬 Missing IDs filled: {set(all_ids) - set(ids)}")


if __name__ == "__main__":
    fill_missing_ids("movies_MISSING.txt", "movies.txt")
