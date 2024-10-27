#!/usr/bin/env zsh

feed_describe_album_score() {
	jq -r '.[] | "\(.image_url)|\(.title)|\(.subtitle)|\(.score)/10"' -r
}

feed_describe_album() {
	jq -r '.[] | "\(.image_url)|\(.title)|\(.subtitle)"' -r
}

feed_img() {
	jq -r '.[] | "\(.image_url)"'
}

feed_favorite_albums() {
	curl -sL 'https://purarue.xyz/feed_api/data/?offset=0&limit=500&order_by=score&sort=desc&ftype=album' | jq
}

feed_albums() {
	curl -sL 'https://purarue.xyz/feed_api/data/?offset=0&limit=100&ftype=album'
}

feed_recent_albums() {
	local recent="${1?:provide recent duration}"
	local after_epoch="$(date -d "-$recent" +%s)"
	curl -sL 'https://purarue.xyz/feed_api/data/?offset=0&limit=100&order_by=score&sort=desc&ftype=album' |
		jq --arg AFTER "${after_epoch}" '.[] | select(.when >= ($AFTER | tonumber))' |
		jq -s
}

feed() {
	feed-cli "$@" | glow -
}

listens-feed() {
	feed -F 'listen' "$@"
}
alias feed-listens=listens-feed
