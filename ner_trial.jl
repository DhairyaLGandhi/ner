"""
	returns a dictionary with words as keys and glove embeddings as vals
"""
function get_vecs()
	fname = "/Users/dhairyagandhi/Downloads/ner/sequence_tagging/data/glove.6B/glove.6B.100d.txt";
	file = open(fname);
	lines = eachline(file);
	dic = Dict()
	for line in lines
	    words = split(line, isspace)
	    arr = Array{Float64}(undef, length(words) - 1)
	    for (ind, i) in enumerate(words[2:end])
	        arr[ind] = parse(Float64, string(i) )
	    end
	    dic[words[1]] = arr
	end

	dic["UNK"] = fill(-1,100)

	return dic
end


"""
	returns vectors with words and tags
"""
function get_words_tags()
	fname = "/Users/dhairyagandhi/Downloads/conll-corpora/conll2000/train.txt";
	f = open(fname);
	lines = eachline(f);
	words = []
	tags = []
	for line in lines
		if length(line) == 0
			push!(words, "")
			push!(tags, "EOS")
			continue
		end
        line = strip(line, ' ')
        t = split(line, isspace)
        push!(words, lowercase(t[1]))
        push!(tags, t[end])
        #break
    end
    return words, tags
end

function get_vocab_(words, tags)
	return Set(words), Set(tags)
end


"""
	takes words from `get_words_tags()` and converts them into an array of sentences.
"""
function make_sents(words)
	sents = []
	sent = []
	for word in words
		if word != ""
			push!(sent, word)
		else
			push!(sents, sent)
			sent = []
		end
	end

	sents
end


function fork(model; n=2)
	forks = []
	for i in 1:n
		f = Chain()
		add(f, model)
		push!(forks, f)
	end
	return forks
end

"""
	takes result of `make_sents()` as input and produces array of padded/ truncated sentences as output
"""
function get_padded_sentences(sents)
	sent_length = 35
	for (i,s) in enumerate(sents)
        if length(s) < sent_length
            push!(s, fill("padder", (1, sent_length - length(s)))...)
        elseif length(s) > sent_length
            sents[i] = s[1:sent_length]
        end
    end
    sents
end

global padder
padder = zeros(Int, 100);

"""
	takes a string as input and produces a vector using glove embeddings
"""
function get_sent_vec(s::String, word_vectors)
	words = split(s, isspace)
	sent = validate_words(words)
	arr = try
		arr = map(x -> word_vectors[lowercase(x)], words)
	catch ex
		arr = []
	end
	arr = hcat(arr...)
	return arr
end

"""
	takes sents and makes training set
"""
function make_train(sents)
	train = []
	for s in sents
		formatted = join(s, " ")
		push!(train, get_sent_vec(formatted))
	end
	return train
end

"""
	check whether words in a sentence exist in given vocab (glove in this case), else assign UNK
"""
function validate_words(sent, vocab = glove_words)
	for (i, word) in enumerate(sent)
		if word in vocab
		else
			sent[i] = "UNK"
		end
	end
	sent
end


"""
	takes dictoinary of glove embeddings, and returns a `Set` of all the words in it
"""
function get_glove_words(word_vectors)
	glove_words = Set(keys(word_vectors))
	glove_words
end
# left = Sequential()
# left.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
#                forget_bias_init='one', return_sequences=True, activation='tanh',
#                inner_activation='sigmoid', input_shape=(99, 13)))
# right = Sequential()
# right.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
#                forget_bias_init='one', return_sequences=True, activation='tanh',
#                inner_activation='sigmoid', input_shape=(99, 13), go_backwards=True))

# model = Sequential()
# model.add(Merge([left, right], mode='sum'))

using Flux
function make_model()
	left = LSTM(100, 64, σ)
	right = LSTM(100, 64, σ)
	bidir(X) = vcat(right(X), left(reverse(X, dims = 2)))
	model = Chain(bidir, 
				Dense(64))

	# rev_X(X) = reverse(X, dims = 2)
end



# function python_stuff()
# open(fname) do f:
#     words, tags = [], []
#     for line in f:
#         line = line.strip()
#         if (len(line) == 0 or line.startswith("-DOCSTART-")):
#             if len(words) != 0:
#                 niter += 1
#                 if self.max_iter is not None and niter > self.max_iter:
#                     break
#                 yield words, tags
#                 words, tags = [], []
#         else:
#             ls = line.split(' ')
#             word, tag = ls[0],ls[-1]
#             if self.processing_word is not None:
#                 word = self.processing_word(word)
#             if self.processing_tag is not None:
#                 tag = self.processing_tag(tag)
#             words += [word]
#             tags += [tag]
# 
# end