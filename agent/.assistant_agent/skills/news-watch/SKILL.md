---
name: news-watch
description: Use this skill to poll news RSS feeds for new articles and post them to Discord.
---

# RSS ニュース定期巡回スキル

RSSフィードを巡回し、前回巡回以降の新着記事だけをDiscordへ投稿する。状態（cursor）はファイルで管理し、巡回のたびにこのスキルの手順に従う。

## 状態ファイル

パス: `/tmp/news_feed_state.json`

媒体ごとに、最後にDiscord投稿が完了した記事の最大公開日時を `cursors` から取得する。日時はJSTのISO 8601形式。

`cursors` がない場合は既存の `posted` から媒体ごとの最大 `published_at` を作る。既存記録もない媒体は、今回RSSにある全記事を投稿候補にする。

## 対象媒体とRSS URL

- NHK: https://www.nhk.or.jp/rss/news/cat0.xml
- 毎日: https://mainichi.jp/rss/etc/mainichi-flash.rss
- 朝日: https://www.asahi.com/rss/asahi/newsheadlines.rdf
- Yahoo!ニュース: https://news.yahoo.co.jp/rss/topics/top-picks.xml
- BBC: https://feeds.bbci.co.uk/news/rss.xml

## 投稿先

Discordチャンネル: `1536093940711686236`

## 手順

### 1. 状態ファイルを読み、cursorを準備する

上記「状態ファイル」の規則に従い、媒体ごとのcursorを用意する。

### 2. web_fetchでRSSを直接取得する

**サブエージェント** を使い、 **`web_fetch` ツールだけ** で対象媒体のRSS URLを直接取得する。Tavily、検索、記事ページ、一般Webページは使わない。

各RSSのXMLから **全entry** を抽出し、各entryについてタイトル・URL・公開日時・RSS内要約本文（`description` / `summary` / `content`等、あれば）を取得する。公開日時がないentryだけをスキップする。公開日時はJSTのISO 8601形式に正規化する。

サブエージェントには、媒体ごとの全entryを必ず構造化して返させる。RSS取得失敗、または公開日時・URLを含む全entryの一覧を返せなかった媒体は、その媒体を失敗として扱う。失敗媒体の記事は投稿せず、そのcursorも更新しない。

### 3. cursorより新しい記事を選ぶ

各媒体について、公開日時がその媒体のcursorより新しい記事だけを投稿候補にする。RSS全体の`lastBuildDate`や`updated`は新着判定に使わない。

### 4. 日本語の要約を作る

RSS内の要約本文がある記事だけ、その内容だけを材料に日本語で1文・60〜100字程度に要約する。RSS内要約がない記事には要約を付けない。記事本文は取得せず、RSS外の情報で要約を補完しない。

### 5. 新着記事をDiscordへ投稿する

候補を上記チャンネルへ投稿する。複数メッセージに分けてよい。各記事は以下の形式にする。

- `HH:MM（JST）｜タイトル（媒体名）`
- 日本語要約（ある場合のみ）
- URL

### 6. 投稿成功後にcursorを更新する

媒体ごとに、その媒体の候補記事をすべてDiscordへ投稿できた場合だけ、cursorを投稿した記事の最大公開日時へ更新する。候補が0件の媒体はcursorを維持する。

RSS取得またはDiscord投稿に失敗した媒体のcursorは更新しない。状態更新後、`posted`などcursor以外の過去記事記録は削除してよい。

### 7. 実行結果を報告する

処理完了後、このチャンネルへ媒体ごとの取得件数・投稿件数・cursor更新の有無・失敗内容（あれば）を簡潔に報告する。
