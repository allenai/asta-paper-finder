define open-url
	if [ "$$(uname)" = "Darwin" ]; then \
		open $(1); \
	elif [ "$$(uname)" = "Linux" ]; then \
		xdg-open $(1); \
	else \
		echo "Unsupported operating system"; \
		exit 1; \
	fi
endef