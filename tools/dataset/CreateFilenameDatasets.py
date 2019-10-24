import sys, getopt, os
import git
sys.path.append(git.Repo('.', search_parent_directories=True).working_tree_dir)


def main(argv):

    # Get folderpath from arg
    input_directory = ''
    output_directory = ''
    pattern = ''

    try:
        opts, args = getopt.getopt(argv,"hi:o:p:")
    except getopt.GetoptError:
        print('Usage -i <input_directory> -o <output_directory> -p <pattern>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h","-help","help","--help"):
            print('Usage -i <input_directory> -o <output_directory>')
            sys.exit()
        elif opt in ("-o", "--out","--output",):
            output_directory = arg
        elif opt in ("-i",):
            input_directory = arg
        elif opt in ("-p", "--pattern"):
            pattern = arg

    


if __name__ == "__main__":
  main(sys.argv[1:])